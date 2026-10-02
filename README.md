# 자원 최적화 — 다음 15분 전력 예측

제6회 K-인공지능 제조데이터 분석 경진대회 과제 ⑤.

```
Dataset/
  raw/               원자료 CSV
  preprocessed/      전처리 결과 + 판정 기록(manifest)
Model/               학습 산출물 — 모델 파일과 results.json
Function/            전처리 + 학습 + 추론 + 자원 최적화
Dashboard/           React 화면 + FastAPI 서버
archive/
  prev_contest/      지난 대회 작업
  this_contest/      이번 대회 모델링 이력 (Modeling · Modeling_RNN · Modeling_GRU_Tree_Ensemble)
```

코드(`Function/`)와 자료(`Dataset/`)·산출물(`Model/`)을 갈라 뒀다.
화면의 API 서버는 `Function/serving.py`를 통해 `Model/`과 `Dataset/preprocessed/`를 읽는다.

## 어디부터 보면 되나

| 하고 싶은 것 | 보는 곳 |
|---|---|
| 화면을 띄운다 | `Dashboard/README.md` |
| 자료를 다시 전처리한다 | `Function/README.md` |
| 모델을 다시 비교한다 | 같은 문서. `python Function/modeling.py` |
| 어떤 모델이 왜 뽑혔는지 | `Model/results.json` |
| 예전에 뭘 했는지 | `archive/this_contest/*/README.md` |

## 대시보드 실행 환경

Python **3.11.17**과 Node.js/npm이 필요하다. macOS arm64에서 Node.js **24.9.0**,
npm **11.6.0**으로 실행을 확인했다. Windows 명령도 아래에 제공하지만 Windows 실행은 아직 검증하지 않았다.
`.python-version`은 버전 관리 도구용 기록이며, 시스템 Python 버전을 자동으로 바꾸지는 않는다.

### macOS / Linux

저장소 루트에서 Python 3.11로 가상환경을 만들고 활성화한다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd Dashboard
npm ci
npm run dev
```

### Windows PowerShell

Python 3.11을 설치한 뒤 저장소 루트에서 실행한다.
활성화 스크립트 없이 가상환경 Python을 직접 호출하므로 PowerShell 실행 정책 변경은 필요 없다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd Dashboard
npm ci
npm run build
..\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8065
```

브라우저에서 <http://127.0.0.1:8065>를 연다. 종료는 서버 터미널에서 `Ctrl+C`.
저장소에 전처리 데이터와 모델이 포함되어 있어 최초 실행에 전처리나 재학습은 필요 없다.
다음 실행에는 macOS/Linux에서 가상환경 활성화 후 `Dashboard`에서 `npm start`를,
Windows에서는 위의 가상환경 Python 서버 명령을 사용한다. 화면 코드를 바꾸면 다시 빌드한다.

### 협업 시 환경 관리

- `.venv/`, `node_modules/`, `Dashboard/dist/`는 각자 생성하며 Git에서 제외한다.
- 루트 `requirements.txt`는 정상 실행한 환경의 전체 패키지 버전을 고정한다.
  하위 폴더의 requirements는 해당 기능의 직접 의존성을 기록한다.
- Python 패키지를 변경하면 루트와 해당 하위 requirements를 함께 갱신하고 실행을 확인한다.
- 프런트엔드 패키지를 변경하면 `package.json`과 `package-lock.json`을 함께 커밋한다.
  팀원은 변경을 내려받은 뒤 `npm ci`로 설치한다.
- 현재 `Model/model_et.joblib`은 **scikit-learn 1.2.2**로 저장됐다.
  다른 버전으로 읽으면 모델 로딩에 실패할 수 있다. 버전을 올릴 때는 같은 버전으로 재학습하고
  모델 파일, 의존성, 평가 결과를 함께 갱신한다.
- 팀원이 변경을 내려받은 뒤에는 가상환경에서 루트 requirements를 다시 설치한다.
  이 파일은 macOS 실행 환경에서 검증했으며, 다른 OS에서 설치·실행 문제가 생기면 공유해 함께 확인한다.

## 지금 결과

전처리 하나를 모든 모델이 공유한다. **평가 행이 같으므로 수치를 나란히 놓을 수 있다.**
공통 24,576행 · 시험 4,991행 · 전진검증 4구간(04/01·05/01·06/01·07/01 각 24일).

| 모델 | 전진검증 | ± | 시험 |
|---|---:|---:|---:|
| **ExtraTrees + GRU 앙상블** | **47.43** | 30.47 | **45.54** |
| ExtraTrees | 49.32 | 34.96 | 54.95 |
| HistGradientBoosting | 57.61 | 35.26 | 60.01 |
| GRU | 60.66 | 28.05 | 49.92 |
| LSTM | 63.16 | 33.99 | 57.66 |
| RandomForest | 91.68 | 61.09 | 120.10 |
| 직전값 유지 (학습 없음) | 205.15 | 10.30 | 180.33 |
| 시각×요일 중앙값 (학습 없음) | 1339.49 | 971.61 | 1540.36 |

**앙상블이 두 기준 모두 1위다.** 다만 전진검증 이득은 작다 — 구간별 이득이
`+5.65 · −1.36 · −4.05 · +7.31` 로 평균 +1.89 인데 표준편차가 4.73 이다.
**4구간으로는 이 이득이 확실하다고 말할 수 없다.** 시험 구간 이득(+9.42)은 크지만
그 구간은 선정에 쓸 수 없다.

GRU 와 LSTM 은 **구별되지 않는다**(차이 평균 +2.50, 표준편차 6.13).

## 규약

- 선정은 **전진검증 평균 MSE 로만** 한다. 시험 구간은 선정에 쓰지 않는다
- 순환신경망은 **시드 3개 예측을 평균**해 보고한다. 시험 성적으로 시드를 고르지 않는다
- 결합 가중치를 고르지 않는다 — 구간마다 최적이 반대 방향이라 고정 반반
- 전처리는 **날짜를 코드에 박지 않는다.** 2022년 자료를 넣어도 돌아간다


## 현재 데이터 흐름

```text
Dataset/raw
    ↓ Function/preprocessing.py
Dataset/preprocessed
    ↓ Function/modeling.py
Model/
    ↓ Function/serving.py + Function/net_infer.py + Function/optimization.py
Dashboard/backend (FastAPI)
    ↓ HTTP API
Dashboard/src (React)
```

전처리와 계산은 `Function/` 한 곳에서 공유한다. 예전 대시보드의 독립 파이프라인은
`archive/dashboard_v1/`에 보관되어 있으며, 현재 화면에서는 사용하지 않는다.

서버는 ExtraTrees 예측과 GRU의 시드 3개 평균 예측을 고정 반반으로 결합한다.
GRU 추론은 별도 프로세스에서 실행한다. GRU 파일이 없거나 사용할 수 없으면
트리 단독으로 동작하며 `/api/meta`에 실제 모델 구성이 표시된다.

현재 화면은 전력 관제와 자원 최적화를 제공한다. 시나리오 API도 있지만 화면에서 직접 호출하지 않는다.
생산량은 전력 예측 모델의 입력이 아니므로, 시나리오에서 생산량을 바꿔도 전력 예측은 바뀌지 않는다.

## 실행 확인

가상환경 Python으로 다음을 실행하면 저장 모델의 추론 결과와 보고된 시험 성능을 비교한다.

```bash
python Function/verify_serving.py
```

macOS arm64에서 Python 3.11.17과 루트 requirements로 앙상블 로딩,
전력 예측·자원 최적화 API 및 브라우저 화면 표시를 확인했다.
