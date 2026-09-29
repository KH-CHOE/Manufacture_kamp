# 6단계: 다음 15분 전력 예측 최종 모델링

처음 보는 사람도 문제 정의부터 전처리, 변수 생성, 모델 비교, 시간 검증, 하이퍼파라미터 탐색, 최종 학습과 오차 분석까지 따라갈 수 있도록 정리한 제출용 패키지입니다.

## 핵심 파일

- `6단계_최종_모델링_보고서.html`: 전체 개발 과정과 수치·그림을 포함한 독립 HTML 보고서.
- `final_input_data.csv`: 최종 모델의 입력 35개와 관리 열 6개를 담은 24,190행 CSV. `split`은 train/test/boundary_excluded.
- `final_model.joblib`: 학습 16,895행에 적합한 최종 ExtraTrees Pipeline. 입력 순서와 선택 설정도 포함.
- `train_model.py`: 최종 입력 CSV에서 5-fold 검증과 동일한 최종 모델 재학습 및 시험 지표 재현.
- `build_report.py`: `evidence/`의 실험 기록과 최종 데이터에서 보고서 생성. 재학습 없음.
- `prepare_final_input.py`: 기존 프로젝트 원본·전처리 CSV와 보조 시간 변수 CSV에서 최종 입력 재생성.
- `predict.py`: 이미 가공된 입력 변수 CSV로 저장 모델 예측.
- `selection.json`: CV에서 확정된 35개 변수, 최종 파라미터 및 선정 규칙.
- `research_history/`: 최종 입력 선정에 사용한 27개 단일 후보·252개 앙상블 탐색 코드와 후보 입력 CSV. 재실행 시 상위 제출 모델은 건드리지 않음.
- `evidence/`: 기준 모델, A–G 입력 묶음, 진단, 두 단계의 탐색, 최종 후보·fold·예측 결과의 원본 CSV/JSON 사본.
- `requirements.txt`: 검증에 사용한 Python 패키지 버전.

## 실행

현재 폴더에서:

```bash
python train_model.py
python build_report.py
python predict.py
```

원본부터 최종 입력을 다시 만들려면 `python prepare_final_input.py`를 먼저 실행합니다. 이 코드는 프로젝트의 `데이터셋/okm_augumented_2021.csv`, `okm_augumented_2021_preprocssed_3.csv`와 이전 작업의 `5단계/output/HS/model_ready.csv`를 읽고, 시간 정렬·정답 일치를 검증합니다. 이미 포함된 `final_input_data.csv`만으로는 학습 코드와 저장 모델을 재현할 수 있습니다.

학습 코드 실행 시 `reproduced_metrics.csv`, `reproduced_folds.csv`, `reproduced_test_predictions.csv`, `reproduction_metadata.json`을 만들고 `final_model.joblib`을 동일 구성으로 재저장합니다. 최종 35개 선택 입력에는 결측이 없으며, Median Imputer는 새 결측 유입에 대비한 Pipeline 구성입니다. 스케일링은 수행하지 않습니다.

예측 시 저장된 모델은 가공된 35개 입력 열을 필요로 합니다. `predict.py`는 시간별 raw CSV를 그대로 입력받는 실시간 예측 서버가 아닙니다.

## 수치와 해석

- 최종 5-fold 검증 MSE 36.4192
- 시험 7,294행 MSE 60.1629, MAE 4.9016, R² 0.9842
- 동일 분할에서 재학습한 이전 29변수 모델의 시험 MSE 66.7987 대비 약 9.9% 감소

최종 평가의 70:30 시간 분할은 기존 실험과 같지만, CV 마지막 검증이 외부 학습 범위를 넘지 않도록 경계 1행을 수정했습니다. 이전 실험들에서 같은 시험 구간을 이미 여러 차례 관찰했으므로, 이 수치를 새로운 독립 기간의 성능으로 주장하지 않습니다. 생산·기상 입력은 예측 시점에 알려진다는 조건입니다.
