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
npm run train:model      # ../Function/modeling.py --models et
```

현재 트리 모델은 **scikit-learn 1.2.2** 형식이다. 환경 버전만 올리면 읽지 못할 수 있으므로
모델과 requirements를 함께 관리한다. 모델·자료 경로는 환경변수 `MODEL_PATH`·`DATA_PATH`로 바꿀 수 있다.

## API

| | |
|---|---|
| `GET /api/meta` | 재생 가능일·모델 정보·시험 구간 성능 |
| `GET /api/snapshot?day&cursor&threshold` | 그 시점의 실측·예측·경보·임계값 |
| `GET·POST /api/optimization` | 그 시간의 최소 인원과 비용 |
| `POST /api/scenario` | 생산량·인원을 바꿔 비용 비교 |

응답 모양은 예전 판과 같다. 프런트엔드(`src/`)는 그대로 가져왔다.

## 알아 둘 것

**시나리오에서 생산량을 바꿔도 전력 예측은 변하지 않는다.** 이 모델은 `생산량` 을
입력으로 쓰지 않는다 — 가이드북 p57 이 생산량을 입력에서 빼 최적화 변수로 돌리고,
실측으로도 넣어서 나아지지 않았다(전진검증 64.230 / 63.646 / 63.409 동률).
응답의 `productionIsInput: false` 와 `productionNote` 가 그 사실을 알린다.
비용은 인원·단가에 따라 정상적으로 달라진다.
지금 화면은 `/api/scenario` 를 부르지 않는다.

**지금 쓰는 모델은 ExtraTrees + GRU 앙상블이다**(전진검증 47.43 · 시험 45.54).
`Model/model_et.joblib` 과 `Model/model_gru.pt` 를 둘 다 읽어 **고정 반반**으로 섞는다.

`model_gru.pt` 가 없거나 예전 형식이면 **트리만으로 서빙하고 그 사실을 알린다**
(ExtraTrees 단독 49.32 / 54.95). 조용히 다른 모델로 바뀌지 않는다.

> **순환신경망 추론은 시드 3개를 모두 돌려 평균한다.** 보고 수치가 시드평균이므로
> 첫 시드만 쓰면 다른 값이 나온다. 표준화 통계도 저장된 것을 쓴다 — 추론 자료로 다시
> 재면 시험 구간을 본 것이 된다.
>
> **순환신경망 추론은 별도 프로세스로 실행한다.** `Function/serving.py`가 현재 Python 환경으로
> `Function/net_infer.py`를 호출해 예측을 받아 온다. sklearn과 torch의 OpenMP 충돌을 피하기 위한 구성이다.
