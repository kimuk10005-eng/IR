
# -*- coding: utf-8 -*-
import os
import json
import csv
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# =========================
# 사용자 설정
# =========================
TRAIN_DIR = r"C:\Users\MASL\Desktop\코드개선\3차_리뉴얼\3rd db"
SAVE_DIR  = r"C:\Users\MASL\Desktop\코드개선\3차_리뉴얼\kumoh_binary_model_save"

# floor는 긴 파일이므로 슬라이싱
FLOOR_WINDOW = 32
FLOOR_STEP = 8

# liquid는 파일 1개 = 이벤트 샘플 1개
LIQUID_BASELINE_TAIL = 20   # baseline 마지막 몇 개를 붙일지
TARGET_LEN = 32             # 모델 입력 길이
MIN_FLOOR_DATA_LEN = FLOOR_WINDOW

EPOCHS = 30
BATCH_SIZE = 16
RANDOM_STATE = 42


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def try_float(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def read_csv_rows(path: str):
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.reader(f))
        except Exception as e:
            last_err = e
    raise last_err


def parse_signal_from_rows(rows):
    """
    반환:
    {
      'baseline_raw': np.array,
      'data_raw': np.array,
      'final_base': float or None
    }
    """
    baseline_vals = []
    data_vals = []
    final_base = None

    for row in rows[1:]:  # header 제외
        if not row:
            continue
        rec_type = row[0].strip().lower() if len(row) > 0 and row[0] else ""
        if rec_type == "baseline":
            v = try_float(row[1] if len(row) > 1 else None)
            if v is None:
                v = try_float(row[3] if len(row) > 3 else None)
            if v is not None:
                baseline_vals.append(v)
        elif rec_type == "data":
            v = try_float(row[3] if len(row) > 3 else None)  # current_raw
            if v is None:
                v = try_float(row[6] if len(row) > 6 else None)
            if v is not None:
                data_vals.append(v)
        elif rec_type == "final_base":
            if len(row) > 1:
                final_base = try_float(row[1])

    return {
        "baseline_raw": np.asarray(baseline_vals, dtype=np.float32),
        "data_raw": np.asarray(data_vals, dtype=np.float32),
        "final_base": final_base,
    }


def normalize_1d(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-6:
        sd = 1.0
    return (x - mu) / sd


def pad_or_crop_1d(x, target_len):
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= target_len:
        return x[-target_len:]
    pad = np.full((target_len - len(x),), x[0] if len(x) > 0 else 0.0, dtype=np.float32)
    return np.concatenate([pad, x], axis=0)


def make_features(seq):
    """
    입력: 1차원 raw sequence
    출력: (T, 4)
      ch0: z-score raw
      ch1: 1차 미분
      ch2: 2차 미분
      ch3: 최근 앞부분 평균 대비 차이
    """
    seq = np.asarray(seq, dtype=np.float32)
    z = normalize_1d(seq)
    d1 = np.gradient(z).astype(np.float32)
    d2 = np.gradient(d1).astype(np.float32)

    front = max(4, len(seq) // 4)
    base_ref = float(np.mean(seq[:front])) if len(seq) > 0 else 0.0
    diff = (seq - base_ref).astype(np.float32)
    diff = normalize_1d(diff)

    feat = np.stack([z, d1, d2, diff], axis=-1).astype(np.float32)
    return feat


def build_floor_samples(floor_dir):
    X, y, names = [], [], []
    file_count = 0
    skip_count = 0

    if not os.path.isdir(floor_dir):
        return X, y, names, file_count, skip_count

    for fn in sorted(os.listdir(floor_dir)):
        if not fn.lower().endswith(".csv"):
            continue
        file_count += 1
        path = os.path.join(floor_dir, fn)
        try:
            rows = read_csv_rows(path)
            parsed = parse_signal_from_rows(rows)
            data_raw = parsed["data_raw"]

            if len(data_raw) < MIN_FLOOR_DATA_LEN:
                skip_count += 1
                continue

            made = 0
            for start in range(0, len(data_raw) - FLOOR_WINDOW + 1, FLOOR_STEP):
                seq = data_raw[start:start + FLOOR_WINDOW]
                feat = make_features(seq)
                X.append(feat)
                y.append(0)  # floor
                names.append(fn)
                made += 1

            if made == 0:
                skip_count += 1

        except Exception:
            skip_count += 1
            continue

    return X, y, names, file_count, skip_count


def build_liquid_samples(liquid_dir):
    X, y, names = [], [], []
    file_count = 0
    skip_count = 0

    if not os.path.isdir(liquid_dir):
        return X, y, names, file_count, skip_count

    for fn in sorted(os.listdir(liquid_dir)):
        if not fn.lower().endswith(".csv"):
            continue
        file_count += 1
        path = os.path.join(liquid_dir, fn)

        try:
            rows = read_csv_rows(path)
            parsed = parse_signal_from_rows(rows)
            baseline_raw = parsed["baseline_raw"]
            data_raw = parsed["data_raw"]

            if len(data_raw) == 0:
                skip_count += 1
                continue

            baseline_tail = baseline_raw[-LIQUID_BASELINE_TAIL:] if len(baseline_raw) > 0 else np.array([], dtype=np.float32)
            seq = np.concatenate([baseline_tail, data_raw], axis=0)
            seq = pad_or_crop_1d(seq, TARGET_LEN)
            feat = make_features(seq)

            X.append(feat)
            y.append(1)  # liquid
            names.append(fn)

        except Exception:
            skip_count += 1
            continue

    return X, y, names, file_count, skip_count


def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(16)),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print(f"[TRAIN_DIR] {TRAIN_DIR}")
    print(f"[SAVE_DIR]  {SAVE_DIR}")
    ensure_dir(SAVE_DIR)

    floor_dir = os.path.join(TRAIN_DIR, "floor")
    liquid_dir = os.path.join(TRAIN_DIR, "liquid")

    X_floor, y_floor, n_floor, floor_files, floor_skip = build_floor_samples(floor_dir)
    X_liq, y_liq, n_liq, liq_files, liq_skip = build_liquid_samples(liquid_dir)

    print("\n[FILE COUNT]")
    print({"floor": floor_files, "liquid": liq_files})
    print("[SAMPLE COUNT]")
    print({"floor": len(X_floor), "liquid": len(X_liq)})
    print("[SKIP COUNT]")
    print({"floor": floor_skip, "liquid": liq_skip})

    X = np.asarray(X_floor + X_liq, dtype=np.float32)
    y = np.asarray(y_floor + y_liq, dtype=np.int32)

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"학습 데이터가 한 클래스만 잡혔습니다. 현재 클래스: {np.unique(y).tolist()}"
        )

    print("\n[CLASS DISTRIBUTION]")
    print(Counter(y.tolist()))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("[CLASS WEIGHT]")
    print(class_weight)

    model = build_model(X.shape[1:])

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, "binary_liquid_floor.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=1
    )

    prob = model.predict(X_val, verbose=0).reshape(-1)
    pred = (prob >= 0.5).astype(np.int32)

    print("\n[VALIDATION REPORT]")
    print(classification_report(
        y_val, pred,
        target_names=["floor", "liquid"],
        digits=4,
        zero_division=0
    ))

    print("[CONFUSION MATRIX]")
    print(confusion_matrix(y_val, pred, labels=[0, 1]))

    config = {
        "target_len": TARGET_LEN,
        "floor_window": FLOOR_WINDOW,
        "floor_step": FLOOR_STEP,
        "liquid_baseline_tail": LIQUID_BASELINE_TAIL,
        "threshold": 0.5,
        "label_map": {"0": "floor", "1": "liquid"},
        "feature_channels": 4
    }
    with open(os.path.join(SAVE_DIR, "binary_liquid_floor_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n완료: 학습 및 모델 저장 종료")


if __name__ == "__main__":
    main()
