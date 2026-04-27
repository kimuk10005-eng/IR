"""260325주 학습 DB 기반 1차원 IR 액체/바닥 이진 분류 학습 스크립트.

핵심 아이디어
- 기존 레포의 CSV 포맷(record_type, baseline_raw, current_raw)을 재사용한다.
- 한 파일을 한 개 시계열로 읽은 뒤, 슬라이딩 윈도우로 샘플을 만든다.
- 파일 경로/이름에 liquid/floor 키워드가 있으면 약한 라벨(weak label)로 사용한다.
- 모델은 Conv1D + BiLSTM + Dense 헤드(= LSTM-CNN 하이브리드).

실행 예시
python experiments/liquid_floor_260325/train_lstm_cnn_liquid_floor.py \
  --data-dir "database/ir data(get)/260325주 학습/trimmed_manual_split" \
  --epochs 80 --window-size 32 --stride 4
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception as exc:  # pragma: no cover
    raise ImportError("scikit-learn 이 필요합니다. `pip install scikit-learn`") from exc

try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        LSTM,
        BatchNormalization,
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        Input,
        MaxPooling1D,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError("tensorflow 가 필요합니다. `pip install tensorflow`") from exc


@dataclass
class SampleWindow:
    x: np.ndarray
    y: int
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1D IR 액체/바닥 LSTM-CNN 학습")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="database/ir data(get)/260325주 학습/trimmed_manual_split",
        help="liquid/, floor/ 하위 폴더가 포함된 데이터 경로",
    )
    parser.add_argument("--save-dir", type=str, default="experiments/liquid_floor_260325/artifacts")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def iter_csv_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.rglob("*.csv")):
        if path.is_file():
            yield path


def infer_binary_label(path: Path) -> Optional[int]:
    lower_name = path.name.lower()
    lower_parent = path.parent.name.lower()

    liquid_hit = ("liquid" in lower_name) or ("liquid" in lower_parent)
    floor_hit = ("floor" in lower_name) or ("floor" in lower_parent)

    if liquid_hit and not floor_hit:
        return 1
    if floor_hit and not liquid_hit:
        return 0
    return None


def _to_float(v: str) -> Optional[float]:
    try:
        return float(str(v).strip())
    except Exception:
        return None


def read_change_signal(path: Path) -> Optional[np.ndarray]:
    """CSV에서 change 시퀀스를 복원한다.

    우선순위
    1) 명시 change 컬럼
    2) current_raw - base(평균 baseline_raw)
    """
    baseline_values: List[float] = []
    row_bases: List[float] = []
    currents: List[float] = []
    explicit_change: List[float] = []

    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    rows: Optional[List[dict]] = None

    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue

    if rows is None:
        return None

    for row in rows:
        record_type = str(row.get("record_type", "")).strip().lower()
        baseline_raw = _to_float(row.get("baseline_raw", ""))
        base_raw = _to_float(row.get("base_raw", ""))
        current_raw = _to_float(row.get("current_raw", ""))
        change = _to_float(row.get("change", ""))

        if record_type == "baseline":
            if baseline_raw is not None:
                baseline_values.append(baseline_raw)
            elif base_raw is not None:
                baseline_values.append(base_raw)
            continue

        if record_type != "data":
            continue

        if change is not None:
            explicit_change.append(abs(change))

        if current_raw is not None:
            currents.append(current_raw)
            if base_raw is not None:
                row_bases.append(base_raw)

    if explicit_change:
        signal = np.asarray(explicit_change, dtype=np.float32)
        return signal if signal.size > 0 else None

    if not currents:
        return None

    if baseline_values:
        base = float(np.mean(baseline_values))
    elif row_bases:
        base = float(np.mean(row_bases))
    else:
        return None

    signal = np.abs(np.asarray(currents, dtype=np.float32) - base)
    return signal if signal.size > 0 else None


def robust_zscore(signal: np.ndarray) -> np.ndarray:
    med = np.median(signal)
    mad = np.median(np.abs(signal - med)) + 1e-6
    return ((signal - med) / (1.4826 * mad)).astype(np.float32)


def to_windows(signal: np.ndarray, y: int, window_size: int, stride: int, src: str) -> List[SampleWindow]:
    out: List[SampleWindow] = []
    if signal.size < window_size:
        return out

    for start in range(0, signal.size - window_size + 1, stride):
        seg = signal[start : start + window_size]
        seg = robust_zscore(seg)
        out.append(SampleWindow(x=seg[:, None], y=y, source_file=src))
    return out


def split_by_file(
    file_names: Sequence[str], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[set, set, set]:
    uniq = sorted(set(file_names))
    rng = random.Random(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = set(uniq[:n_train])
    val_files = set(uniq[n_train : n_train + n_val])
    test_files = set(uniq[n_train + n_val :])
    return train_files, val_files, test_files


def build_model(window_size: int, lr: float) -> Model:
    inputs = Input(shape=(window_size, 1), name="ir_window")

    x = Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(LSTM(48, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid", name="is_liquid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ir_lstm_cnn_binary")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def make_xy(samples: Sequence[SampleWindow]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x = np.asarray([s.x for s in samples], dtype=np.float32)
    y = np.asarray([s.y for s in samples], dtype=np.float32)
    f = [s.source_file for s in samples]
    return x, y, f


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_windows: List[SampleWindow] = []
    used_files = 0

    for csv_path in iter_csv_files(data_dir):
        y = infer_binary_label(csv_path)
        if y is None:
            continue
        signal = read_change_signal(csv_path)
        if signal is None:
            continue

        windows = to_windows(
            signal=signal,
            y=y,
            window_size=args.window_size,
            stride=args.stride,
            src=str(csv_path),
        )
        if not windows:
            continue

        all_windows.extend(windows)
        used_files += 1

    if not all_windows:
        raise RuntimeError("유효한 윈도우 샘플이 없습니다. --data-dir 및 CSV 형식을 확인하세요.")

    x_all, y_all, file_names = make_xy(all_windows)
    train_files, val_files, test_files = split_by_file(
        file_names, args.train_ratio, args.val_ratio, args.seed
    )

    train_samples = [s for s in all_windows if s.source_file in train_files]
    val_samples = [s for s in all_windows if s.source_file in val_files]
    test_samples = [s for s in all_windows if s.source_file in test_files]

    x_train, y_train, _ = make_xy(train_samples)
    x_val, y_val, _ = make_xy(val_samples)
    x_test, y_test, _ = make_xy(test_samples)

    model = build_model(args.window_size, args.learning_rate)

    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    pred_prob = model.predict(x_test, verbose=0).reshape(-1)
    pred = (pred_prob >= 0.5).astype(np.int32)

    report = classification_report(
        y_true=y_test.astype(np.int32),
        y_pred=pred,
        target_names=["floor", "liquid"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test.astype(np.int32), pred, labels=[0, 1]).tolist()

    model.save(save_dir / "model_lstm_cnn_binary.keras")
    (save_dir / "metrics.json").write_text(
        json.dumps(
            {
                "used_files": used_files,
                "num_windows": int(len(all_windows)),
                "train_windows": int(len(train_samples)),
                "val_windows": int(len(val_samples)),
                "test_windows": int(len(test_samples)),
                "confusion_matrix": cm,
                "classification_report": report,
                "final_val_auc": float(history.history["val_auc"][-1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== 학습 완료 ===")
    print(f"사용 파일 수: {used_files}")
    print(f"윈도우 수(train/val/test): {len(train_samples)}/{len(val_samples)}/{len(test_samples)}")
    print("혼동행렬 [floor, liquid]:")
    print(np.asarray(cm))


if __name__ == "__main__":
    run(parse_args())
