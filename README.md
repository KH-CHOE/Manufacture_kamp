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
│   ├── model_et.joblib             최종 ExtraTrees 모델 (기상 미사용)
│   ├── model_gru.pt                최종 GRU 모델 및 표준화 정보
│   ├── results.json               최종·구성 모델 성능 및 결합 비율
│   └── manifest.json              모델·데이터·환경의 일치 검사 정보
├── Function/
│   ├── settings.py                 경로·입력·학습 설정
│   ├── data_preprocessing.py       원본 전처리 및 입력 변수 생성
│   ├── model_training.py           ExtraTrees·GRU 학습·평가·결합
│   ├── model_validation.py         저장 모델 일치 검사
│   ├── power_prediction.py         모델 로드·전력 예측·조회
│   ├── gru_prediction.py           GRU 추론 전용 프로세스
│   ├── staffing_cost.py            인원·비용 계산
│   ├── ai_chat.py                  대화체 요약·OpenAI 연결
│   ├── holiday_calendar.json         2021~2026년 전국 공휴일
│   └── electricity_tariff.json     전력 요금·부하 시간대
└── Dashboard/
    ├── backend/app.py             FastAPI API·정적 화면 제공
    ├── src/                       React·TypeScript 화면
    ├── public/fonts/              글꼴 및 LICENSE.txt
    ├── index.html                 화면 진입점
    ├── build.mjs                  프론트엔드 빌드
    ├── tsconfig.json              TypeScript 설정
    ├── package.json               npm 실행 명령
    └── package-lock.json          JavaScript 의존성 버전
```

과거 작업 이력과 검증 전용 스크립트는 제출 구성에서 제외했습니다. 모델 파일의 지문·환경·입력 일치를 검사하는 `model_validation.py`는 실제 실행에 필요하므로 포함합니다.

## 3. 데이터와 실행 흐름

```text
Dataset/raw → Function/data_preprocessing.py → Dataset/preprocessed
                                               ↓
                                Function/model_training.py → Model
                                               ↓
                             Function/power_prediction.py (모델 추론)
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
python Function/data_preprocessing.py
python Function/model_training.py --models et,gru,ensemble
```

위 명령은 최종 구성인 ExtraTrees·GRU·앙상블만 학습합니다. 비교 실험 스크립트와 RF·HGB·LSTM·기준선 비교 학습 코드는 제출 구성에서 제외했습니다.

재학습은 시간이 오래 걸리며 `Model/`의 모델·평가 결과·파일 지문을 갱신합니다. 실행용 모델을 임의로 섞어 교체하지 마세요. 전체 학습은 구간·시드별 완료 결과를 재사용합니다.

### 전처리 규칙

- 한 시간의 네 전력 열을 15분 간격으로 펼치고 다음 구간 전력을 정답으로 만듭니다.
- 시간은 0~23의 정수이며 날짜 내 중복이 없어야 합니다. 비정상 시간이 있는 날짜는 통째로 제외합니다. 정상 시간이 있는 불완전 날짜는 유지합니다.
- 결측은 해당 시점 이전 관측만의 누적 중앙값으로 처리합니다. 시간 공백을 넘는 시차·이동 통계·신경망 입력 창은 사용하지 않습니다.
- 전력을 역산할 수 있는 파생 열과 달력 정보를 다시 적은 열은 모델 입력에서 제외합니다.
- 생산량·기상 같은 시간당 정보는 한 시간 늦춘 변수로 생성하며 동일 시간의 미확정 원값을 모델 입력에 넣지 않습니다.
- 처리 결과와 이유는 `Dataset/preprocessed/processed_manifest.json`에 남깁니다.

2021~2026년 한국 데이터는 `Function/holiday_calendar.json`에서 해당 연도의 공휴일·대체공휴일·전국 선거일·임시공휴일을 자동 적용합니다. 여러 연도가 섞인 자료도 지원합니다. 등록되지 않은 연도는 오류로 중단하며, `--no-calendar`를 명시한 경우에만 주말 기준으로 처리합니다. JSON에는 공식 출처와 확인 날짜를 함께 기록했습니다. 회사별 휴무일은 별도이며 2021~2025년 근로자의 날은 이 전국 공휴일 목록에 포함하지 않습니다.

2021년은 저장 모델 재현을 위해 `settings.py`의 분할 날짜 20210725를 사용합니다. 다른 연도는 기본적으로 뒤쪽 30%를 시험으로 나누며, `--split-date YYYYMMDD`로 변경할 수 있습니다. 공휴일 파일에는 분할 날짜를 저장하지 않습니다.

다른 자료의 경로와 달력은 다음처럼 지정할 수 있습니다. 해당 자료의 형식·열 이름은 `Function/settings.py`의 스키마와 맞아야 합니다.

```bash
python Function/data_preprocessing.py --raw 새자료.csv --out 정리자료.csv --calendar Function/holiday_calendar.json
```

## 6. 모델 구성과 성능

현재 화면은 **ExtraTrees 0.47 + GRU 0.53**의 결합 예측을 사용합니다. ExtraTrees는 과거 전력·달력 변수를, GRU는 과거 96개 구간(하루)의 전력과 달력 변수를 사용합니다. 최종 모델 입력에 기상 변수는 포함하지 않습니다.

기상 열은 전처리·화면의 맥락 정보와 기존 모델의 공통 평가 행을 재현하는 유효성 검사에 사용합니다. 현재 학습·추론 경로에는 기온·풍속·습도·강수량의 지연 열이 필요합니다. 자원 최적화에는 원본의 `생산량`·`공장인원` 열도 필요합니다.

저장된 결과 기준 공통 평가 대상은 24,192행, 시험은 4,991행, 전진검증은 4구간입니다. 아래 지표는 MSE이며 낮을수록 좋습니다.

| 모델 | 전진검증 평균 | 구간 간 표준편차 | 시험 |
|---|---:|---:|---:|
| ExtraTrees + GRU (트리 0.47) | 37.2850 | 18.2540 | 46.5607 |
| ExtraTrees | 38.73 | 21.67 | 54.99 |
| GRU | 48.22 | 17.92 | 53.00 |

상세 결과와 결합 비율 탐색은 `Model/results.json`에 있습니다. 현재 제출에는 최종 앙상블과 두 구성 모델의 결과만 포함합니다. 모델마다 입력과 학습 기간이 다르므로 성능 차이를 알고리즘 자체의 우열로 단정하지 않습니다.

**평가 해석:** 결합 비율은 시험 구간 MSE를 최소화하도록 선택했습니다. 따라서 결합 시험 성능은 비율 선택에 사용된 값이며 독립적인 최종 평가가 아닙니다. 새 외부 데이터에 대한 성능 보장은 아닙니다.

## 7. 자원 최적화의 계산 방식

현재 화면은 과거 기록 재생용입니다. 정시 직전 45분 예측을 사용해 해당 정시부터 한 시간의 생산 목표·인원·비용을 표시합니다. 생산 목표는 그 정시의 기록된 생산 실적에서 자동 조회합니다.

- **인원:** 대상 날짜보다 앞선 기록 중 같은 시간·평일/휴일 유형이고 목표 이상을 생산한 사례의 파생 인원을 올림하여 최소값을 선택합니다. 일치 사례가 없으면 미산정으로 표시합니다.
- **생산 이익:** 시간당 생산 목표 × 개당 생산 이익.
- **인건비:** 참고 인원 × 시간당 시급. 22:00~06:00에는 1.5배를 적용합니다.
- **전력량 요금:** 정시 예측 전력 × 1시간 × 계절·부하별 단가.
- **기본요금 배분:** 작년 피크 전력 × 기본요금 단가 ÷ 해당 월 일수 ÷ 24.
- **시간당 영업 이익:** 생산 이익 − 인건비 − 전력량 요금 − 기본요금 배분.

원자료의 `공장인원`은 생산량과 전력에서 계산한 파생값이며 실제 배치 인원을 뜻하지 않습니다. 화면의 추천은 과거 사례 기반 참고값입니다. 전력비는 15분 예측이 한 시간 유지된다는 가정이며 2021년 요금표를 사용합니다. 실제 청구서 전체 항목과 세금을 재현하지 않습니다.

공휴일 판정은 전처리와 동일한 다년도 달력을 사용하지만 전력 요금 단가는 여전히 2021년 기준입니다. 다른 연도의 실제 비용 계산에는 요금표 갱신이 필요합니다.

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

모델·전처리 데이터 경로는 `MODEL_PATH`·`DATA_PATH` 환경변수로 지정할 수 있습니다. 로딩 시 모델 파일 지문·입력 설정·scikit-learn/PyTorch 버전을 검사하고, 트리와 GRU의 추론 행이 일치하는지 확인합니다. 데이터 파일 지문은 학습 기록에 보관하며 로딩 시 자동 비교하지 않습니다. 자원 최적화의 원본 조회 경로는 `settings.py`의 `RAW_DEFAULT`를 사용하므로 새 자료로 실행할 때 원본과 전처리 자료를 함께 맞춰야 합니다.

## 9. 최종 실행 검증

2026-10-06~07에 위 macOS 환경에서 다음 항목을 확인했습니다.

- 원본의 기본 전처리 결과: **24,477행 × 46열**, 제출된 `processed.csv`와 SHA-256 일치.
- 기본 설정 그대로 ExtraTrees·GRU 전체 재학습 완료: 전진검증 4구간·최종 시험 구간, GRU 각 3시드(총 15조합). 시험 MSE **46.5607**, 트리 비중 **0.47** 재현.
- 새 학습 산출물의 모델 로드·결합 추론·비용 계산·챗 요약 정상 실행. 추론 MSE **46.560719**로 학습 기록과 일치.
- 제출 모델·평가 결과의 파일 지문 일치 확인. 재학습 검증은 별도 폴더에서 진행하고 기존 제출 모델 파일을 유지.
- `requirements.txt` 설치와 `python -m pip check` 통과.
- `npm run build`의 TypeScript 검사·프론트엔드 빌드 통과.
- API의 전력 조회·생산 실적 재생·비용 계산·대화체 요약 정상 응답. 위험 상태, 잘못된 입력, 없는 날짜 및 API 키 누락 응답 확인.
- 브라우저에서 날짜 전환·자원 최적화·피크 입력 적용·알림 창·요약 갱신 확인, 콘솔 오류 없음.
- 실제 OpenAI SDK를 모의 HTTP 전송에 연결해 대화체 요청·질의응답 도구 호출·응답 해석·음성 바이트 처리 확인.

실제 OpenAI API 키를 사용한 외부 질의응답·음성 생성은 검증하지 않았습니다. 해당 기능의 최종 응답은 API 연결·키 권한·계정 상태에 따라 달라집니다. 검증용 코드와 학습 중간 산출물은 제출 저장소 밖에서 실행했습니다.

## 10. 글꼴 및 라이선스

화면은 우아한형제들의 **배민 한나체 Air(BM HANNA Air)**를 사용합니다. 글꼴 라이선스 전문은 `Dashboard/public/fonts/LICENSE.txt`에 포함되어 있습니다. 공식 안내: https://www.woowahan.com/fonts
