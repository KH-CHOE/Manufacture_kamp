# 후보 탐색 재현

`python search_candidates.py`를 실행하면 공통 24,190행에서 27개 단일 후보와 252개 앙상블 조합을 다시 검증하고 이 폴더에 결과를 저장합니다. 실행에는 수 분이 걸릴 수 있습니다. 이미 저장된 결과는 상위 `evidence/`에 있습니다. 상위의 제출용 `final_model.joblib`은 수정하지 않습니다.

입력 `candidate_input_data.csv`는 최종 선정 이전의 BG 29개와 추가 후보 17개를 포함합니다. `data_manifest.json`은 raw·전처리 입력의 출처와 열 목록을 기록합니다.
