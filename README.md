# 제조공장 전력 예측 및 자원 최적화 대시보드

제6회 K-인공지능 제조데이터 분석 경진대회 과제 ⑤의 제출 프로젝트입니다. 제공된 제조공장 데이터를 전처리하고 다음 15분의 전력을 예측하며, 과거 기록을 재생하는 대시보드에서 전력 추이·피크 경보·인원 및 비용 계산을 확인할 수 있습니다.

## 1. 주요 기능

- **전력 관제:** 실제 전력과 다음 15분 예측, 날짜·시점 선택 및 재생, CSV 내보내기.
- **피크 경보:** 재생 시점까지 관측된 올해 최대 피크, 예측 오차를 반영한 참고 알림 기준, 직접 설정한 기준의 초과 경고.
- **자원 최적화:** 대상 정시의 과거 생산 실적을 자동 조회하고, 이전 날짜의 같은 시간·날 유형 실적에서 목표 이상을 달성한 참고 인원과 비용을 계산.
- **AI 챗:** API 키 없이 수치 기반 대화체 요약 제공. OpenAI API 키 입력 시 모델의 설명·질의응답·음성 기능 제공.

이 화면은 제공된 **2021년 데이터의 과거 기록 재생**입니다. 실시간 센서 수집이나 설비 자동 제어 기능은 포함하지 않습니다.

## 2. 폴더 구조

```text
Git/ 또는 저장소 루트/
├── README.md                       전체 설명 및 실행 안내
├── requirements.txt                Python 환경 버전
├── .python-version                 Python 버전 기록
├── Dataset/
│   ├── raw/okm_augumented_2021.csv  제공 원본 데이터
│   └── preprocessed/
│       ├── processed.csv           현재 모델 입력 데이터
│       └── processed_manifest.json 전처리 규칙·처리 결과·파일 지문
├── Model/
│   ├── model_et.joblib             ExtraTrees 모델
│   ├── model_et_wx.joblib          기상 포함 ExtraTrees 비교 모델
│   ├── model_hgb.joblib            HistGradientBoosting 비교 모델
│   ├── model_gru.pt                GRU 모델 및 표준화 정보
│   ├── model_lstm.pt               LSTM 비교 모델
│   ├── results.json               모델별 성능 및 결합 비율
│   ├── manifest.json              모델·데이터·환경의 일치 검사 정보
│   ├── duplicate_evaluation.json   반복 날짜 제외 보조 평가
│   └── weather_blend_comparison.json 기상 포함 여부 비교
├── Function/
│   ├── config.py                  경로·변수·학습 설정
│   ├── preprocessing.py           원본 전처리 및 입력 변수 생성
│   ├── modeling.py                모델 학습·평가·저장
│   ├── artifacts.py               파일 지문과 저장 모델 일치 검사
│   ├── evaluate_duplicates.py     학습에 포함된 보조 평가
│   ├── compare_weather_blend.py   기상 포함·미포함 결합 비교
│   ├── serving.py                 모델 로드·예측·조회
│   ├── net_infer.py               신경망 추론 전용 프로세스
│   ├── optimization.py            인원·비용 계산
│   ├── assistant.py               대화체 요약·OpenAI 연결
│   ├── calendar_2021.json         공휴일·분할 설정
│   └── tariff.json                전력 요금·부하 시간대
└── Dashboard/
    ├── backend/app.py             FastAPI API·정적 화면 제공
    ├── backend/requirements.txt   서버 직접 의존성
    ├── src/                       React·TypeScript 화면
    ├── public/fonts/              글꼴 및 LICENSE.txt
    ├── index.html                 화면 진입점
    ├── build.mjs                  프론트엔드 빌드
    ├── tsconfig.json              TypeScript 설정
    ├── package.json               npm 실행 명령
    └── package-lock.json          JavaScript 의존성 버전
```

과거 작업 이력과 검증 전용 스크립트는 제출 구성에서 제외했습니다. 모델 파일의 지문·환경·입력 일치를 검사하는 `artifacts.py`는 실제 실행에 필요하므로 포함합니다.

## 3. 데이터와 실행 흐름

```text
Dataset/raw → Function/preprocessing.py → Dataset/preprocessed
                                               ↓
                                Function/modeling.py → Model
                                               ↓
                             Function/serving.py (모델 추론)
                                               ↓
                            Dashboard/backend/app.py (API)
                                               ↓
                                  Dashboard/src (화면)
```

최초 실행에는 저장된 전처리 데이터와 모델을 사용합니다. 원본 전처리와 재학습은 자동 실행되지 않습니다. 자원 최적화는 원본의 시간당 생산 실적과 파생 인원 자료도 조회합니다.

## 4. 대시보드 실행

### 환경

Python **3.11.17**, Node.js와 npm이 필요합니다. macOS arm64에서 Node.js 24.9.0·npm 11.6.0으로 실행한 환경을 기준으로 합니다. 저장 모델은 **scikit-learn 1.2.2**에서 생성되었으므로 루트 `requirements.txt`의 버전을 사용하세요. Windows 명령은 제공하지만 Windows 실행 검증은 하지 않았습니다.

### macOS / Linux

저장소 루트에서 실행합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd Dashboard
npm ci
npm run dev
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd Dashboard
npm ci
npm run build
..\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8065
```

브라우저에서 **http://127.0.0.1:8065/** 를 엽니다. 종료하려면 서버 터미널에서 `Ctrl+C`를 누릅니다. 다음 실행에는 가상환경을 활성화한 뒤 `Dashboard`에서 `npm start`를 사용합니다. 프론트엔드 코드를 변경하면 `npm run build` 후 새로고침하세요.

`.venv/`, `node_modules/`, `Dashboard/dist/`, Python 캐시는 실행 중 생성되는 로컬 파일이며 제출 소스에 포함할 필요가 없습니다. `.git/`는 로컬 저장소 관리 정보입니다. 폴더를 압축 제출할 때 이 항목과 `.env`·API 키를 제외하세요.

## 5. 전처리와 재학습

가상환경을 활성화한 후 저장소 루트에서 실행합니다.

```bash
python Function/preprocessing.py
python Function/modeling.py --models baseline,rf,et,hgb,et_wx,gru,lstm,ensemble
```

재학습은 시간이 오래 걸리며 `Model/`의 모델·평가 결과·파일 지문을 갱신합니다. 실행용 모델을 임의로 섞어 교체하지 마세요. 전체 학습은 구간·시드별 완료 결과를 재사용합니다.

### 전처리 규칙

- 한 시간의 네 전력 열을 15분 간격으로 펼치고 다음 구간 전력을 정답으로 만듭니다.
- 시간은 0~23의 정수이며 날짜 내 중복이 없어야 합니다. 비정상 시간이 있는 날짜는 통째로 제외합니다. 정상 시간이 있는 불완전 날짜는 유지합니다.
- 결측은 해당 시점 이전 관측만의 누적 중앙값으로 처리합니다. 시간 공백을 넘는 시차·이동 통계·신경망 입력 창은 사용하지 않습니다.
- 전력을 역산할 수 있는 파생 열과 달력 정보를 다시 적은 열은 모델 입력에서 제외합니다.
- 생산량·기상 같은 시간당 정보는 한 시간 늦춘 변수로 생성하며 동일 시간의 미확정 원값을 모델 입력에 넣지 않습니다.
- 처리 결과와 이유는 `Dataset/preprocessed/processed_manifest.json`에 남깁니다.

다른 자료의 경로와 달력은 다음처럼 지정할 수 있습니다. 해당 자료의 형식·열 이름은 `Function/config.py`의 스키마와 맞아야 합니다.

```bash
python Function/preprocessing.py --raw 새자료.csv --out 정리자료.csv --calendar Function/calendar_2022.json
```

## 6. 모델 구성과 성능

현재 화면은 **ExtraTrees 0.47 + GRU 0.53**의 결합 예측을 사용합니다. ExtraTrees는 과거 전력·달력 변수를, GRU는 과거 96개 구간(하루)의 전력과 달력 변수를 사용합니다. 기상 포함 ExtraTrees는 비교 후보로 보관하며 현재 결합에는 사용하지 않습니다.

저장된 결과 기준 공통 평가 대상은 24,192행, 시험은 4,991행, 전진검증은 4구간입니다. 아래 지표는 MSE이며 낮을수록 좋습니다.

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

상세 결과는 `Model/results.json`, 기상 변수 비교는 `Model/weather_blend_comparison.json`에 있습니다. 모델마다 입력과 학습 기간이 다르므로 성능 차이를 알고리즘 자체의 우열로 단정하지 않습니다.

**평가 해석:** 결합 비율은 시험 구간 MSE를 최소화하도록 선택했습니다. 따라서 결합 시험 성능은 비율 선택에 사용된 값이며 독립적인 최종 평가가 아닙니다. 반복 날짜 제외 평가는 중복 전력곡선의 영향을 확인하기 위한 보조 결과이고 일부 구간의 날짜 수가 적습니다. 새 외부 데이터에 대한 성능 보장은 아닙니다.

## 7. 자원 최적화의 계산 방식

현재 화면은 과거 기록 재생용입니다. 정시 직전 45분 예측을 사용해 해당 정시부터 한 시간의 생산 목표·인원·비용을 표시합니다. 생산 목표는 그 정시의 기록된 생산 실적에서 자동 조회합니다.

- **인원:** 대상 날짜보다 앞선 기록 중 같은 시간·평일/휴일 유형이고 목표 이상을 생산한 사례의 파생 인원을 올림하여 최소값을 선택합니다. 일치 사례가 없으면 미산정으로 표시합니다.
- **생산 이익:** 시간당 생산 목표 × 개당 생산 이익.
- **인건비:** 참고 인원 × 시간당 시급. 22:00~06:00에는 1.5배를 적용합니다.
- **전력량 요금:** 정시 예측 전력 × 1시간 × 계절·부하별 단가.
- **기본요금 배분:** 작년 피크 전력 × 기본요금 단가 ÷ 해당 월 일수 ÷ 24.
- **시간당 영업 이익:** 생산 이익 − 인건비 − 전력량 요금 − 기본요금 배분.

원자료의 `공장인원`은 생산량과 전력에서 계산한 파생값이며 실제 배치 인원을 뜻하지 않습니다. 화면의 추천은 과거 사례 기반 참고값입니다. 전력비는 15분 예측이 한 시간 유지된다는 가정이며 2021년 요금표를 사용합니다. 실제 청구서 전체 항목과 세금을 재현하지 않습니다.

개당 이익·시급·피크 전력의 ‘기본값’ 체크와 요금표 적용값은 브라우저에 저장할 수 있습니다.

## 8. AI 챗 및 주요 API

API 키가 없으면 실제 조회 수치를 정해진 대화체 문장에 넣어 요약합니다. OpenAI API 키가 있으면 수치·위험 판단 자료를 모델에 전달해 설명과 질문 답변을 생성합니다. 키는 브라우저 저장소에 저장되고 서버에는 요청 헤더로 전달됩니다. 서버는 키를 파일에 저장하지 않습니다. 외부 API 사용에는 네트워크 연결과 해당 계정의 이용 비용이 발생할 수 있습니다.

| API | 용도 |
|---|---|
| `GET /api/meta` | 재생 날짜·모델 정보 |
| `GET /api/snapshot` | 선택 시점의 전력·예측·경보 |
| `POST /api/optimization/replay` | 과거 생산 목표 자동 조회 및 비용 계산 |
| `GET·POST /api/optimization` | 계획 생산량을 직접 지정하는 계산 API |
| `POST /api/scenario` | 생산량·인원 시나리오 비교 |
| `POST /api/briefing` | 전력 상태 설명 |
| `POST /api/chat` | 자료 조회를 통한 질문 답변 |
| `POST /api/speech` | OpenAI 음성 생성 |

모델·데이터 경로는 `MODEL_PATH`·`DATA_PATH` 환경변수로 지정할 수 있습니다. 현재 저장 모델은 파일 지문과 데이터·설정 일치 검사를 거치므로 임의 파일 교체 시 관련 저장 정보도 맞춰야 합니다.

## 9. 글꼴 및 라이선스

화면은 우아한형제들의 **배민 한나체 Air(BM HANNA Air)**를 사용합니다. 글꼴 라이선스 전문은 `Dashboard/public/fonts/LICENSE.txt`에 포함되어 있습니다. 공식 안내: https://www.woowahan.com/fonts
