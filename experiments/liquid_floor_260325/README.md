# liquid_floor_260325 사용 가이드

질문하신 내용 기준으로 핵심만 먼저 정리하면:

- 생성된 위치
  - 학습 스크립트: `experiments/liquid_floor_260325/train_lstm_cnn_liquid_floor.py`
  - 브레인스토밍 문서: `experiments/liquid_floor_260325/brainstorm_plan_ko.md`
- 학습 결과(모델/지표) 저장 위치
  - 기본값: `experiments/liquid_floor_260325/artifacts`

---

## 1) 무엇을 학습하나요?

- 입력: 1차원 IR 시계열 CSV (`record_type`, `baseline_raw`, `current_raw`, `change` 포맷)
- 분류: `floor(0)` vs `liquid(1)` 이진 분류
- 모델: Conv1D + BiLSTM 기반 LSTM-CNN 하이브리드

데이터 기본 경로는 아래로 잡혀 있습니다.

- `database/ir data(get)/260325주 학습/trimmed_manual_split`
  - `liquid/`
  - `floor/`

---

## 2) 실행 전 준비

아래 라이브러리가 필요합니다.

```bash
pip install numpy scikit-learn tensorflow
```

> 참고: 현재 일부 실행 환경에는 tensorflow/scikit-learn이 없어서 학습이 바로 안 될 수 있습니다.

---

## 3) 학습 실행 방법

레포 루트(`/workspace/IR`)에서:

```bash
python experiments/liquid_floor_260325/train_lstm_cnn_liquid_floor.py \
  --data-dir "database/ir data(get)/260325주 학습/trimmed_manual_split" \
  --save-dir "experiments/liquid_floor_260325/artifacts" \
  --epochs 80 \
  --window-size 32 \
  --stride 4
```

---

## 4) 결과는 어디에 생기나요?

기본 `--save-dir` 기준 아래 파일이 생성됩니다.

- `model_lstm_cnn_binary.keras` : 학습된 모델
- `metrics.json` : confusion matrix, classification report, 윈도우 수, 마지막 val_auc 등

즉, 기본 실행이면 여기에 생성됩니다:

- `experiments/liquid_floor_260325/artifacts/model_lstm_cnn_binary.keras`
- `experiments/liquid_floor_260325/artifacts/metrics.json`

---

## 5) 빠른 점검 체크리스트

1. `--data-dir` 경로에 `liquid/`, `floor/` csv가 있는지 확인
2. 파일명이 `_liquid_` 또는 `_floor_` 규칙을 따르는지 확인
3. 라이브러리 설치 확인 (`python -c "import tensorflow, sklearn, numpy"`)
4. 학습 후 `artifacts/` 파일 생성 확인

