# 260325주 학습 DB로 액체 vs 바닥 분류 브레인스토밍

## 1) 문제 정의
- 목표: **1차원 IR 시계열**에서 `액체(liquid)`와 `바닥(floor)`를 이진 분류.
- 데이터 소스: `database/ir data(get)/260325주 학습/trimmed_manual_split/{liquid,floor}`.
- 모델: **CNN + LSTM 하이브리드** (국소 패턴 + 시간적 문맥 동시 반영).

## 2) 기존 코드 재사용 포인트
- 기존 레포 코드에서 사용하던 CSV 포맷(`record_type`, `baseline_raw`, `current_raw`, `change`) 파싱 로직 유지.
- baseline을 활용해 `abs(current_raw - base)`를 복원하는 fallback 유지.
- 파일명 기반 라벨링(`_liquid_`, `_floor_`) 방식을 초기 weak supervision으로 유지.

## 3) 데이터 설계 아이디어
1. **윈도우 분할 학습**
   - 한 CSV를 여러 윈도우로 잘라 표본 확장.
   - 권장 시작값: `window_size=32`, `stride=4`.
2. **파일 단위 분할**
   - 같은 원본 파일에서 나온 윈도우가 train/test에 동시에 들어가면 누수 발생.
   - 따라서 split은 윈도우가 아니라 **파일 단위**로 진행.
3. **정규화**
   - 각 윈도우별 robust z-score(중앙값+MAD) 적용.
   - 조도 변화(밝음/어둠)나 바닥 반사 편차에 강인.

## 4) 모델 구조 아이디어
- `Conv1D(32) -> Conv1D(64)`로 국소 파형 특징 추출.
- `BiLSTM(48)`로 시간 축 문맥 추적.
- `GlobalAveragePooling -> Dense -> Sigmoid`로 최종 이진 판별.
- 지표: Accuracy + AUC (특히 불균형 시 AUC 중점).

## 5) 실험 로드맵
- 실험 A: 현재 기본 하이퍼파라미터(빠른 베이스라인).
- 실험 B: `window_size` 스윕(24/32/48).
- 실험 C: `stride` 스윕(2/4/8)으로 데이터량/일반화 균형.
- 실험 D: floor 종류별 leave-one-floor-out 검증(도메인 일반화).

## 6) 배포/추론 아이디어
- 출력 확률 `p(liquid)`에 대해
  - `p >= 0.6`: 액체
  - `p <= 0.4`: 바닥
  - 중간 구간: 보류/재측정
- 연속 프레임에서 moving vote 적용해 튀는 예측 완화.

## 7) 이번 커밋 산출물
- `train_lstm_cnn_liquid_floor.py`
  - 260325 데이터셋 바로 학습 가능한 end-to-end 스크립트.
  - 모델/메트릭을 `experiments/liquid_floor_260325/artifacts`에 저장.
