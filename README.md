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

공통 24,576행 · 시험 4,991행 · 전진검증 4구간. 수치는 MSE이며 낮을수록 좋다.

> **전처리 변경 후 재학습 전 (2026-10-05).** `시간` 열이 깨진 날(2021-07-13·07-15)을 행 순서로
> 복구하던 것을 **삭제**로 바꿨다. 아래 표와 `Model/` 은 복구 방식 자료로 학습한 결과다.
> 저장 모델의 시험 예측은 새 전처리에서도 같다(결합 47.3819 · 4,991행) — 시험 구간이 삭제한 날과
> 떨어져 있기 때문이다. 공통행은 24,192행, 7월 전진검증 구간은 2,304 → 1,920행으로 줄어
> **전진검증 수치는 재학습해야 새 전처리와 맞는다.** 그 전까지 `verify_serving.py` 는 자료 불일치로 실패한다.

| 모델 | 전진검증 평균 | 구간 간 표준편차 | 반복 날짜 제외 검증 | 시험 |
|---|---:|---:|---:|---:|
| ExtraTrees + GRU | 46.70 | 30.78 | 48.44 | 47.38 |
| ExtraTrees | 49.32 | 34.96 | 53.94 | 54.95 |
| RandomForest | 56.86 | 40.79 | 62.91 | 66.90 |
| HistGradientBoosting | 57.61 | 35.26 | 60.31 | 60.01 |
| GRU | 58.24 | 29.53 | 55.35 | 54.97 |
| LSTM | 63.39 | 32.81 | 59.05 | 60.55 |
| 직전값 유지 | 205.15 | 10.30 | 156.88 | 180.33 |
| 시각×요일 중앙값 | 1339.49 | 971.61 | 1363.72 | 1540.36 |

**이번 설정에서는 ExtraTrees + GRU가 전진검증 평균 기준 1위다.** 반복 날짜를 제외한 보조 평가에서도 1위를 유지한다. 시험은 선정 기준으로 쓰지 않았다.

학습에 있던 것과 완전히 같은 하루 전력 곡선이 5월 검증 24일 중 13일, 6월 24일 중 23일에 반복된다. 이를 제외하면 6월은 하루만 남는다. 4월·7월 검증과 시험에서는 이 기준의 반복 날짜가 없다. 불완전한 하루와 근사 중복은 검사하지 않았다.

RF의 표본 재추출 설정을 고치자 전진검증 MSE가 기존 91.68에서 56.86으로 개선됐다. 기존 RF 비교는 불리한 설정의 영향을 받았다. GRU·LSTM은 입력 창과 표준화 오류를 고쳐 재학습한 결과다.

트리와 신경망은 입력 정보와 학습 기간이 다르고, 검증 구간도 네 개뿐이다. 작은 차이를 알고리즘의 확실한 우열로 해석하지 않는다. 이번 시험 점수는 이미 검토한 시험 자료에서 오류를 고친 재평가이며, 새로운 외부 자료 검증은 아니다.

전체 기록: [results.json](Model/results.json) · [반복 날짜 제외 평가](Model/duplicate_evaluation.json) · [수정·검증 내역](REVIEW_20261003.md)

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
GRU 추론은 별도 프로세스에서 실행한다. GRU 파일이 없으면 트리 단독으로 동작하며 `/api/meta`에 실제 모델 구성이 표시된다.
파일이 있는데 잘못됐거나 추론에 실패하면 서버 시작을 중단한다.

현재 화면은 전력 관제와 인원·비용 가정을 제공한다. 계획 생산량은 사용자가 입력하며, 미래 실적을 자동으로 사용하지 않는다. 원자료의 공장인원은 생산량/전력 합으로 만든 파생값이므로 실제 최소 인원의 근거로 해석하지 않는다. 시나리오 API도 있지만 화면에서 직접 호출하지 않는다.
생산량은 전력 예측 모델의 입력이 아니므로, 시나리오에서 생산량을 바꿔도 전력 예측은 바뀌지 않는다.

## 실행 확인

가상환경 Python으로 다음을 실행하면 저장 모델의 추론 결과와 보고된 시험 성능을 비교한다.

```bash
python Function/verify_pipeline.py
python Function/verify_regressions.py
python Function/verify_training.py
python Function/verify_serving.py
```

macOS arm64에서 Python 3.11.17과 루트 requirements로 앙상블 로딩,
전력 예측·자원 최적화 API 및 브라우저 화면 표시를 확인했다.

이번 수정 모델의 학습·검증 Python은 3.11.7이며, 학습 라이브러리는 루트 requirements의 고정 버전과 같다.
기존 3.11.17 실행 안내와 `.python-version`은 유지한다. 정확한 학습 버전과 파일 지문은 `Model/manifest.json`에 기록한다.
