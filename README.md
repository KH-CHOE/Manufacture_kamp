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

공통 24,192행 · 시험 4,991행 · 전진검증 4구간. 수치는 MSE이며 낮을수록 좋다.
(2026-10-06 재학습 — `시간` 열이 깨진 날 삭제 · 기상 포함 ExtraTrees 후보 · 결합 비율 탐색 반영)

| 모델 | 전진검증 평균 | 구간 간 표준편차 | 반복 날짜 제외 검증 | 시험 |
|---|---:|---:|---:|---:|
| ExtraTrees + GRU (트리 0.47) | 37.28 | 18.25 | 38.80 | 46.56 |
| ExtraTrees + 기상 (비교 후보) | 38.02 | 21.52 | 42.58 | 54.17 |
| ExtraTrees | 38.73 | 21.67 | 43.35 | 54.99 |
| RandomForest | 44.78 | 26.37 | 50.82 | 67.44 |
| HistGradientBoosting | 46.39 | 22.30 | 49.09 | 59.39 |
| GRU | 48.22 | 17.92 | 45.33 | 53.00 |
| LSTM | 51.74 | 17.96 | 47.40 | 59.64 |
| 직전값 유지 | 199.67 | 12.81 | 151.40 | 180.33 |
| 시각×요일 중앙값 | 1313.08 | 994.86 | 1337.32 | 1540.06 |

**결합(ExtraTrees + GRU)이 전진검증·시험·반복 날짜 제외 검증 모두에서 1위다.**

**결합 비율 — 시험 구간 기준으로 골랐다(팀 결정).** 트리 비중을 0.01~0.99 로 훑어 시험 MSE 가 가장 낮은
**0.47**(GRU 0.53)를 쓴다. **시험이 고르는 데 쓰였으므로 결합의 시험 MSE 46.56 는 낙관적이다.**
고르는 데 쓰지 않은 값은 전진검증(37.29)이다. 참고로 고정 0.5 는 전진 36.98 · 시험 46.59,
전진검증 기준으로 골랐다면 0.68(전진 36.11 · 시험 47.91)였다.
기준은 `Function/config.py` 의 `BLEND_SELECT` 로 바꿀 수 있다(`"test"` · `"forward"`).

**기상 포함 ExtraTrees 는 비교 후보다.** 결합에는 기상을 쓰지 않는 ExtraTrees 를 넣는다. 이번 재학습에서는 기상 포함판이
전진검증 38.02 · 시험 54.17 로 미포함판(38.73 · 54.99)보다 조금 낮다.
기상은 한 시간 전 확정값만 쓴다(정각 관측인지 시간 평균인지 원자료로 알 수 없어서).

**전진검증 수치가 이전(공통 24,576행)보다 크게 낮아진 것은 행 기준이 바뀐 탓이다.** `시간` 열이 깨진 2021-07-13·07-15 를
삭제하면서 7월 구간에서 7/13~7/16 나흘분 행이 빠졌다(2,304 → 1,920행). 7월 구간 ExtraTrees MSE 가 100.73 → 58.36 으로
내려간 것이 평균을 끌어내렸다(빠진 날들의 오차를 따로 재지는 않았다 — 학습 자료도 함께 바뀌었다). 이전 수치와 나란히 비교하지 않는다.
시험 구간 4,991행은 같다.

학습에 있던 것과 완전히 같은 하루 전력 곡선이 5월 검증 24일 중 13일, 6월 24일 중 23일에 반복된다. 이를 제외하면 6월은 하루만 남는다.
4월·7월 검증과 시험에서는 이 기준의 반복 날짜가 없다. 불완전한 하루와 근사 중복은 검사하지 않았다.

트리와 신경망은 입력 정보와 학습 기간이 다르고, 검증 구간도 네 개뿐이다. 작은 차이를 알고리즘의 확실한 우열로 해석하지 않는다.
이번 시험 점수는 이미 검토한 시험 자료에서 오류를 고친 재평가이며, 새로운 외부 자료 검증은 아니다.

전체 기록: [results.json](Model/results.json) · [반복 날짜 제외 평가](Model/duplicate_evaluation.json) · [수정·검증 내역](REVIEW_20261003.md)

## 규약

- 모델 순위는 **전진검증 평균 MSE** 로 매긴다
- 순환신경망은 **시드 3개 예측을 평균**해 보고한다. 시험 성적으로 시드를 고르지 않는다
- 결합 비율은 **시험 MSE 최소**로 고른다(트리 비중 0.01~0.99, 팀 결정). 그래서 결합의 시험 성적은 낙관적이다
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

서버는 ExtraTrees(기상 미사용) 예측과 GRU의 시드 3개 평균 예측을 `Model/manifest.json` 의 `blend_weight`(시험 구간 기준으로 고른 트리 비중)로 결합한다.
대시보드의 **전력 브리핑**은 화면에서 입력한 OpenAI API 키로 ChatGPT 브리핑·질의응답을 한다(키는 저장하지 않는다). `Dashboard/README.md` 참조.
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
