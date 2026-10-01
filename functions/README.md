# functions

화면이 가져다 쓰는 계산 부분.

| 폴더 | 무엇 | 상태 |
|---|---|---|
| `modeling/` | 전처리 + 모델 비교. 원자료 → 전처리 CSV → 모델·결과 | 있음 |
| `optimization/` | 생산 목표·인원·비용 계산 | **아직 없다.** 지금은 `Dashboard/optimization/` 에 있다 |

## modeling 은 두 단계다

```
datasets/raw/*.csv  ──preprocessing.py──▶  datasets/preprocessed/processed.csv
                                                      │
                                           modeling.py ──▶ results.json + 모델 파일
```

`preprocessing.py` 가 **전처리**, `modeling.py` 가 **모델링**이다.
둘은 설정(`config.py`)을 공유하므로 한 폴더에 둔다 — 나누면 import 가 꼬인다.

자세한 것은 `modeling/README.md`.
