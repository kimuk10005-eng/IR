import numpy as np
import os
import glob
import pickle
import random
import csv
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# =========================== 설정 ===========================
DATA_DIR = r"C:\Users\MASL\Desktop\ir data(get)\260322"
SAVE_DIR = "./kumoh_lstm_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 15
CHANGE_THRESHOLD = 5.0

# False: CSV에 저장된 change 값을 그대로 사용
# True : abs(current_raw - baseline)로 다시 계산
RECALC_CHANGE_FROM_BASELINE = True

# ✅ 라벨(출력 클래스): 바닥 / 액체 2클래스
CLASSES = ['바닥', '액체']

LIQUID_ALIASES = {
    'water': '물',
    'coffee': '커피',
    'cola': '콜라',
    'milk': '우유',
    'mango': '망고',
    'oil': '기름',
    'matcha': '말차',
    'tomato': '토마토',
    'watermelon': '수박'
}

FLOOR_KEYWORDS = {
    '검대': ['검대'],
    '회대': ['회대'],
    '황대': ['황대'],
    '207회바': ['greyfloor', '회색바닥'],
    '흰책상': ['white', '하양', '흰', 'whitedesk'],
    '나타': ['나타'],
    '회타': ['회타'],
}

REVERSE_DIRECTION_FLOORS = {'검대'}


def detect_liquid_from_filename(filename: str):
    lower = filename.lower()
    for key, kor in LIQUID_ALIASES.items():
        if key in lower:
            return kor
    for kor in LIQUID_ALIASES.values():
        if kor in filename:
            return kor
    return None


def to_float(x):
    try:
        s = str(x).strip().replace('﻿', '')
        if s == '':
            return None
        return float(s)
    except Exception:
        return None


def read_new_format_csv(fp):
    final_base = None
    currents = []
    stored_changes = []
    row_bases = []

    for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
        try:
            with open(fp, 'r', encoding=enc, errors='ignore', newline='') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                names = [str(x).strip().replace('﻿', '') for x in reader.fieldnames]
                if 'record_type' not in names:
                    continue

                for row in reader:
                    rec = (row.get('record_type') or '').strip().lower()
                    if rec == 'final_base' and final_base is None:
                        final_base = to_float(row.get('base_raw'))
                    elif rec == 'data':
                        current_raw = to_float(row.get('current_raw'))
                        base_raw = to_float(row.get('base_raw'))
                        change = to_float(row.get('change'))
                        if current_raw is None:
                            continue
                        currents.append(current_raw)
                        row_bases.append(base_raw)
                        stored_changes.append(change)

                if len(currents) < WINDOW_SIZE:
                    return None, None, None

                base = final_base
                if base is None:
                    valid_row_bases = [b for b in row_bases if b is not None]
                    if not valid_row_bases:
                        return None, None, None
                    base = float(np.mean(valid_row_bases))

                currents = np.array(currents, dtype=np.float32)
                if RECALC_CHANGE_FROM_BASELINE:
                    changes = np.abs(currents - base).astype(np.float32)
                else:
                    fixed_changes = []
                    for i, ch in enumerate(stored_changes):
                        if ch is None:
                            fallback_base = row_bases[i] if row_bases[i] is not None else base
                            ch = abs(currents[i] - fallback_base)
                        fixed_changes.append(ch)
                    changes = np.array(fixed_changes, dtype=np.float32)

                return float(base), currents, changes
        except Exception:
            continue
    return None, None, None


def read_old_format_csv(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    for enc in encodings:
        try:
            final_base = None
            currents = []
            row_bases = []
            stored_changes = []

            with open(fp, 'r', encoding=enc, errors='ignore') as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 2:
                        continue

                    if final_base is None and len(parts) >= 5 and parts[0] == 'Final' and parts[1] == 'Base':
                        final_base = to_float(parts[4])
                        continue

                    if parts[0] == 'Time':
                        current_raw = None
                        base_raw = None
                        change = None

                        for i, p in enumerate(parts):
                            if p == 'BaseRaw' and i + 1 < len(parts):
                                base_raw = to_float(parts[i + 1])
                            elif p == 'CurrentRaw' and i + 1 < len(parts):
                                current_raw = to_float(parts[i + 1])
                            elif p == 'Change' and i + 1 < len(parts):
                                change = to_float(parts[i + 1])

                        if current_raw is not None:
                            currents.append(current_raw)
                            row_bases.append(base_raw)
                            stored_changes.append(change)

            if len(currents) < WINDOW_SIZE:
                continue

            if final_base is not None:
                base = final_base
            else:
                valid_row_bases = [b for b in row_bases if b is not None]
                if not valid_row_bases:
                    continue
                base = float(np.mean(valid_row_bases))

            currents = np.array(currents, dtype=np.float32)
            if RECALC_CHANGE_FROM_BASELINE:
                changes = np.abs(currents - base).astype(np.float32)
            else:
                fixed_changes = []
                for i, ch in enumerate(stored_changes):
                    if ch is None:
                        fallback_base = row_bases[i] if row_bases[i] is not None else base
                        ch = abs(currents[i] - fallback_base)
                    fixed_changes.append(ch)
                changes = np.array(fixed_changes, dtype=np.float32)

            return float(base), currents, changes
        except Exception:
            continue

    return None, None, None


# 새 CSV 형식 우선, 실패하면 예전 형식 fallback

def read_csv_change(fp):
    base, currents, changes = read_new_format_csv(fp)
    if base is not None:
        return base, currents, changes
    return read_old_format_csv(fp)


def create_window_with_label(signal, changes, start, base, floor_type, liquid):
    if start + WINDOW_SIZE > len(signal):
        return None, None

    sig = signal[start:start + WINDOW_SIZE]
    chg = changes[start:start + WINDOW_SIZE]

    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])

    base_eff = base if len(signal) < 20 else 0.7 * base + 0.3 * np.mean(signal[:20])
    ctx_start = max(0, start - 5)
    ctx_end = min(len(signal), start + WINDOW_SIZE + 5)
    ctx_mean = np.mean(signal[ctx_start:ctx_end])

    diff = base_eff - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - base_eff
    direction = np.clip(diff / 300.0, -1.0, 1.0)
    direction_ch = np.full(WINDOW_SIZE, direction)

    window = np.stack([norm, d1, d2, direction_ch])

    # 2클래스 방식: change threshold 이상이면 액체, 아니면 바닥
    label = '액체' if np.max(np.abs(chg)) >= CHANGE_THRESHOLD else '바닥'
    return window, label


def detect_floor_type(filename):
    lower = filename.lower()
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None


def train_models(data_dir=DATA_DIR, save_dir=SAVE_DIR):
    global encoder
    encoder = LabelEncoder().fit(CLASSES)

    floor_files = {f: [] for f in FLOOR_KEYWORDS}
    for fp in glob.glob(os.path.join(data_dir, '*.csv')):
        name = os.path.basename(fp).lower()
        if any(skip in name for skip in ['바닥', 'base']):
            continue

        floor = detect_floor_type(name)
        if floor:
            floor_files[floor].append(fp)

    for floor, file_list in floor_files.items():
        if not file_list:
            continue

        X_all, y_all = [], []

        for fp in file_list:
            liquid = detect_liquid_from_filename(os.path.basename(fp))
            if not liquid:
                continue

            base, signal, changes = read_csv_change(fp)
            if base is None:
                continue

            max_start = len(signal) - WINDOW_SIZE
            if max_start <= 0:
                continue

            step = 3

            for start in range(0, max_start + 1, step):
                win, label = create_window_with_label(signal, changes, start, base, floor, liquid)
                if win is None:
                    continue

                for _ in range(2):
                    aug = win + np.random.normal(0, random.uniform(0.01, 0.05), win.shape)
                    aug[:3] *= random.uniform(0.9, 1.1)
                    X_all.append(aug)
                    y_all.append(label)

        if len(X_all) == 0:
            print(f'[SKIP] {floor}: 학습 데이터가 없습니다. (파일명/파싱/임계값 확인)')
            continue

        X = np.array(X_all).transpose(0, 2, 1)
        y = encoder.transform(y_all)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(enumerate(class_weights))

        inputs = Input(shape=(WINDOW_SIZE, 4))
        x = Conv1D(64, 5, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Conv1D(96, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

        x = Conv1D(128, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.3)(x)

        x = Bidirectional(LSTM(80, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
        x = Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2))(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.4)(x)

        outputs = Dense(len(CLASSES), activation='softmax')(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            X, y,
            epochs=200,
            batch_size=16,
            validation_split=0.15,
            class_weight=class_weight_dict,
            callbacks=[
                EarlyStopping(patience=30, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=12),
                ModelCheckpoint(
                    os.path.join(save_dir, f'model_{floor}.keras'),
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ],
            verbose=1
        )

    pickle.dump(encoder, open(os.path.join(save_dir, 'encoder.pkl'), 'wb'))


if __name__ == '__main__':
    train_models()
    print(f'학습 완료! 모델 저장 위치: {SAVE_DIR}')
