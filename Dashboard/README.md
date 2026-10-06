# 전력 관제·자원 최적화 화면

**계산은 이 폴더에 없다.** 전처리·모델·자원 최적화는 전부 `../Function/` 에 있고,
여기는 HTTP 로 감싸 화면에 내보내는 일만 한다.

```
Dataset/raw ──▶ Function/preprocessing.py ──▶ Dataset/preprocessed
                                                      │
                            Function/modeling.py ──▶ Model/
                                                      │
    Dashboard/backend/app.py ──▶ Function/serving.py ─┘
                 │                      └─▶ Function/optimization.py
                 ▼
            Dashboard/src   (React)
```

## 왜 이렇게 바꿨나

예전 판(`archive/dashboard_v1/`)은 폴더 하나로 돌아가게 만들려고 **안에 모든 것을 복사해
두었다** — `data/`·`models/`·`pipeline/`·`optimization/` 가 전부 들어 있었다.
그래서 같은 로직이 두 벌이 됐고, 화면 쪽 전처리와 모델링 쪽 전처리가 **서로 달랐다.**

| | 예전 `Dashboard/pipeline/` | 지금 `Function/` |
|---|---|---|
| 다른 해 자료 | 6,168행이 아니면 예외로 멈춤 | **돌아간다** |
| 결측 대치 | (날짜, 시간)별로 적어 둔 값 | 엄격히 과거만 보는 누적 중앙값 |
| 깨진 날 | `07-13`·`07-15` 를 날짜로 지정해 삭제 | 규칙으로 복구, 안 되면 그 날만 제외 |
| 공휴일 | `optimization/config.json` 에 따로 | `Function/calendar_2021.json` 한 곳 |

지금은 화면도 모델링도 **같은 전처리 결과 한 장**을 본다.

## 실행

Python **3.11.17**을 사용하고, 먼저 [루트 README의 환경 설정](../README.md#대시보드-실행-환경)을 따른다.
전체 Python 패키지는 저장소 루트 `requirements.txt`로 설치한다.
가상환경을 만들기만 하면 활성화되는 것은 아니다. 아래 npm 명령은 활성화된 환경의 `python3`를 사용한다.

macOS/Linux에서 저장소 루트 기준:

```bash
source .venv/bin/activate
cd Dashboard
npm ci
npm run dev              # 빌드 + 서버. http://127.0.0.1:8065
```

Windows PowerShell 명령은 루트 README에 있다.
저장소에 `Dataset/preprocessed/processed.csv`, `Model/model_et.joblib`, `Model/model_gru.pt`가
포함되어 있으므로 최초 실행에는 전처리나 재학습이 필요 없다.

자료나 모델을 다시 만들 때만, 가상환경을 활성화한 뒤 이 폴더에서 실행한다.

```bash
npm run prepare:data     # ../Function/preprocessing.py
npm run train:model      # 전체 모델 재학습·비교·결합 결과 갱신
```

현재 트리 모델은 **scikit-learn 1.2.2** 형식이다. 환경 버전만 올리면 읽지 못할 수 있으므로
모델과 requirements를 함께 관리한다. 모델·자료 경로는 환경변수 `MODEL_PATH`·`DATA_PATH`로 바꿀 수 있다.

## API

| | |
|---|---|
| `GET /api/meta` | 재생 가능일·모델 정보·시험 구간 성능 |
| `GET /api/snapshot?day&cursor&threshold` | 그 시점의 실측·예측·경보·임계값 |
| `GET·POST /api/optimization` | 계획 생산량에 따른 참고 인원·비용 가정 |
| `POST /api/scenario` | 생산량·인원을 바꿔 비용 비교 |
| `POST /api/briefing` | 재생 시점 전력 브리핑. 헤더 `X-OpenAI-Key` 가 있으면 ChatGPT, 없으면 숫자 요약 |
| `POST /api/chat` | 15분 최대치 관리·공정 질의응답(도구로 재생 시점 자료 조회). 키 필수 |

`/api/optimization`은 `planned_production`(해당 시간의 계획 생산량)을 필수로 받는다.
미래 실적을 자동 사용하지 않으며, 화면에서 계획을 입력한 뒤 적용한다.

## 전력 브리핑 (ChatGPT)

전력 관제 탭 아래의 **전력 브리핑** 구역이다. 계산은 `Function/assistant.py` 에 있다.

- **키** — 화면에서 OpenAI API 키를 입력한다. 키는 **이 브라우저(localStorage)에만** 저장되고
  요청 헤더 `X-OpenAI-Key` 로만 서버에 간다. 서버는 저장·기록하지 않고, 오류 응답에도 담지 않는다.
  키가 없으면 언어 모델을 부르지 않고 숫자 요약만 보여 준다.
- **갱신** — 재생 시각이 한 칸(15분) 넘어갈 때마다 자동 갱신(끄고 켤 수 있음) + **브리핑 갱신** 버튼.
  요청이 진행 중이면 새 요청을 겹쳐 보내지 않고, 끝나면 가장 최근 칸으로 한 번만 다시 부른다.
- **시각 표기** — 재생 행의 현재 구간 [t, t+15분) 은 t+15분에 확정되고, 예측은 다음 구간이다.
  브리핑은 **예측 구간이 끝나기 5분 전** 으로 표기한다. 예) t=10:45 → 11:00까지 확정 · 11:00~11:15 예측 →
  "11:10 브리핑". 재생 자료에 :10 시점 관측은 없으므로 본문에 "○○:○○까지 확정값"을 함께 적는다.
- **하네스** — 시스템 지시문에 공정(볼트·너트, 변압기 총부하 계측), 예측 모델 구성·성능·한계
  (`Model/results.json` 에서 읽음), **관리 목표 — 15분 단위 사용량의 최대치를 높지 않게 계속 유지**,
  피크 관리 원칙(설비 자동 정지 지시 금지, 실측 우선, 숫자 지어내기 금지)을 넣는다.
  요금 산정은 계약 조건에 따라 복잡해 다루지 않는다 — 금액 질문에는 계산하지 않는다고 답한다.
  **숫자는 서버가 계산한다** — 언어 모델은 그 숫자를 읽고 설명·권고만 한다.
- **도구** — 질의응답에서 모델이 부른다. 재생 시점 이후 자료는 주지 않는다.
  `get_briefing_facts` · `get_today_profile` · `get_model_info`
- **모델** — 환경변수 `OPENAI_MODEL`(기본 `gpt-4o-mini`). 요청 본문 `model` 로 바꿀 수도 있다.
- **음성** — 브라우저 내장 음성(Web Speech API)으로 읽는다. 서버·키·요금이 필요 없다.
  **읽어 주기**(다시 누르면 멈춤)와 **위험 시 음성**(기본 꺼짐 — 새 브리핑이 피크 위험일 때만 자동으로 읽음) 스위치가 있다.
  시각은 "15시 15분부터 15시 30분까지", kW 는 "킬로와트"로 바꿔 읽는다. 목소리는 브라우저·운영체제에 따라 다르다 —
  발표에 쓸 컴퓨터에서 미리 들어 볼 것(윈도우는 한국어 음성이 없으면 다른 언어 목소리로 읽을 수 있다).
- **비용 주의** — 연속 재생 중 자동 갱신을 켜 두면 칸마다 호출이 일어난다. 발표 시연 외에는 끄고 버튼을 쓰자.

## 알아 둘 것

**시나리오에서 생산량을 바꿔도 전력 예측은 변하지 않는다.** 이 모델은 `생산량` 을
입력으로 쓰지 않는다 — 가이드북 p57 이 생산량을 입력에서 빼 최적화 변수로 돌리고,
실측으로도 넣어서 나아지지 않았다(전진검증 64.230 / 63.646 / 63.409 동률).
응답의 `productionIsInput: false` 와 `productionNote` 가 그 사실을 알린다.
비용은 인원·단가에 따라 정상적으로 달라진다.
지금 화면은 `/api/scenario` 를 부르지 않는다.

**기본 화면은 ExtraTrees(기상 미사용) + GRU 결합 예측을 사용한다.** 현재 점수는 `Model/results.json`에서 확인한다.
`Model/model_et.joblib` 과 `Model/model_gru.pt` 를 둘 다 읽어 `Model/manifest.json` 의 `blend_weight`
(트리 비중 — 시험 구간 MSE 최소로 고른 값, 팀 결정) 로 섞는다. 기록이 없으면 0.5.
모델 정보 창에 결합 비율과 비교 후보(기상 포함 ExtraTrees 등) 성적이 나온다.

모델의 파일 지문과 scikit-learn 버전을 `Model/manifest.json`과 대조한다.
존재하는 GRU 파일이 잘못됐거나 추론에 실패하면 서버 시작을 중단한다.
GRU 파일 자체가 없으면 트리만 사용하며 `/api/meta`에 실제 구성을 표시한다.

> **순환신경망 추론은 시드 3개를 모두 돌려 평균한다.** 보고 수치가 시드평균이므로
> 첫 시드만 쓰면 다른 값이 나온다. 표준화 통계도 저장된 것을 쓴다 — 추론 자료로 다시
> 재면 시험 구간을 본 것이 된다.
>
> **순환신경망 추론은 별도 프로세스로 실행한다.** `Function/serving.py`가 현재 Python 환경으로
> `Function/net_infer.py`를 호출해 예측을 받아 온다. sklearn과 torch의 OpenMP 충돌을 피하기 위한 구성이다.

인원은 실제 배치 인원이 아니라 원자료의 `생산량÷전력 합`에서 만든 참고 가정이다.
필요한 최소 인원이라고 해석하지 않는다. 전력 비용도 다음 15분 예측이 한 시간 유지된다는 가정이다.
