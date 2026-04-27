# -*- coding: utf-8 -*-
"""
2진 분류 학습 코드: 바닥 / 액체

루트:
C:\Users\MASL\Desktop\3차리뉴얼db

입력 구조:
- trimmed_manual_split/liquid : 사람이 액체라고 잘라둔 구간
- trimmed_manual_split/floor  : 사람이 바닥이라고 남긴 구간

출력:
- kumoh_binary_model_save/model_{floor}.keras
- kumoh_binary_model_save/encoder.pkl
"""

import os
import glob
import csv
import pickle
import random
import re
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

ROOT_DIR = r"C:\Users\MASL\Desktop\3차리뉴얼db"
SPLIT_ROOT = os.path.join(ROOT_DIR, "trimmed_manual_split")
LIQUID_DIR = os.path.join(SPLIT_ROOT, "liquid")
FLOOR_DIR = os.path.join(SPLIT_ROOT, "floor")
SAVE_DIR = os.path.join(ROOT_DIR, "kumoh_binary_model_save")
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 15
SMOOTH_K = 5

LIQUID_BOOST = 4
FLOOR_BOOST = 2
LIQUID_STRIDE = 1
FLOOR_STRIDE = 2

CLASSES = ['바닥', '액체']

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

RAW_RE = re.compile(
    r"Time=(?P<time>[-+]?\d+(?:\.\d+)?)s\s*\|\s*"
    r"BaseRaw=(?P<base>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"CurrentRaw=(?P<current>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"Change=(?P<change>[-+]?\d+(?:\.\d+)?)"
)

def to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None

def smooth_signal(x, k=SMOOTH_K):
    if len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(np.asarray(x, dtype=np.float32), kernel, mode='same')

def remove_spikes(signal, k=3.0, local=5):
    sig = np.array(signal, dtype=np.float32).copy()
    if len(sig) < 3:
        return sig
    for i in range(1, len(sig) - 1):
        left = max(0, i - local)
        right = min(len(sig), i + local + 1)
        local_med = float(np.median(sig[left:right]))
        local_std = float(np.std(sig[left:right])) + 1e-6
        if (
            abs(float(sig[i]) - local_med) > k * local_std
            and abs(float(sig[i]) - float(sig[i - 1])) > 1.2 * local_std
            and abs(float(sig[i]) - float(sig[i + 1])) > 1.2 * local_std
        ):
            sig[i] = np.float32(0.5 * (sig[i - 1] + sig[i + 1]))
    return sig

def detect_floor_type(filename: str):
    lower = filename.lower()
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None

def read_csv_recalc_change(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    def parse_new_format(lines):
        baseline_values, currents, row_bases = [], [], []
        reader = csv.DictReader(lines)
        fields = [str(name).strip() for name in (reader.fieldnames or [])]
        if 'record_type' not in fields:
            return None, None, None
        for row in reader:
            record_type = str(row.get('record_type', '')).strip().lower()
            baseline_raw = to_float(row.get('baseline_raw', ''))
            base_raw = to_float(row.get('base_raw', ''))
            current_raw = to_float(row.get('current_raw', ''))
            raw_data = str(row.get('raw_data', '')).strip()

            if record_type == 'baseline':
                if baseline_raw is not None:
                    baseline_values.append(baseline_raw)
                elif base_raw is not None:
                    baseline_values.append(base_raw)
                continue

            if record_type == 'data':
                if current_raw is None and raw_data:
                    m = RAW_RE.search(raw_data)
                    if m:
                        current_raw = float(m.group('current'))
                        if base_raw is None:
                            base_raw = float(m.group('base'))
                if current_raw is not None:
                    currents.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)

        if len(currents) < WINDOW_SIZE:
            return None, None, None

        if baseline_values:
            base = float(np.mean(baseline_values))
        else:
            valid_row_bases = [b for b in row_bases if b is not None]
            if not valid_row_bases:
                return None, None, None
            base = float(np.mean(valid_row_bases))

        currents = np.array(currents, dtype=np.float32)
        changes = np.abs(currents - base).astype(np.float32)
        return base, currents, changes

    def parse_old_format(lines):
        baseline_values, currents, row_bases = [], [], []
        final_base = None

        for raw_line in lines:
            line = str(raw_line).strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if not parts:
                continue

            if parts[0].startswith('Collecting baseline'):
                if len(parts) >= 3 and parts[1] == 'RawAvg':
                    v = to_float(parts[2])
                    if v is not None:
                        baseline_values.append(v)
                continue

            if len(parts) >= 5 and parts[:4] == ['Final', 'Base', 'Raw', 'Average']:
                v = to_float(parts[4])
                if v is not None:
                    final_base = v
                continue

            if parts[0] == 'Time':
                base_raw = None
                current_raw = None
                for i in range(len(parts) - 1):
                    if parts[i] == 'BaseRaw':
                        base_raw = to_float(parts[i + 1])
                    elif parts[i] == 'CurrentRaw':
                        current_raw = to_float(parts[i + 1])
                if current_raw is not None:
                    currents.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)

        if len(currents) < WINDOW_SIZE:
            return None, None, None

        if final_base is not None:
            base = float(final_base)
        elif baseline_values:
            base = float(np.mean(baseline_values))
        else:
            valid_row_bases = [b for b in row_bases if b is not None]
            if not valid_row_bases:
                return None, None, None
            base = float(np.mean(valid_row_bases))

        currents = np.array(currents, dtype=np.float32)
        changes = np.abs(currents - base).astype(np.float32)
        return base, currents, changes

    for enc in encodings:
        try:
            with open(fp, 'r', encoding=enc, errors='ignore', newline='') as f:
                lines = f.readlines()
            result = parse_new_format(lines)
            if result[0] is not None:
                return result
            result = parse_old_format(lines)
            if result[0] is not None:
                return result
        except Exception:
            continue
    return None, None, None

def create_window(signal, start, base, floor_type):
    if start + WINDOW_SIZE > len(signal):
        return None

    sig = signal[start:start + WINDOW_SIZE]
    norm = (sig - sig.mean()) / (sig.std() + 1e-8)

    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])

    base_eff = base if len(signal) < 20 else 0.7 * base + 0.3 * np.mean(signal[:20])
    ctx_start = max(0, start - 5)
    ctx_end = min(len(signal), start + WINDOW_SIZE + 5)
    ctx_mean = np.mean(signal[ctx_start:ctx_end])

    direction_raw = base_eff - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - base_eff
    direction = np.clip(direction_raw / 300.0, -1.0, 1.0)
    direction_ch = np.full(WINDOW_SIZE, direction)

    return np.stack([norm, d1, d2, direction_ch])

def build_model():
    inp = Input(shape=(WINDOW_SIZE, 4))
    x = Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    out = Dense(len(CLASSES), activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def collect_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, '*.csv')))

def train_models():
    print('=' * 90)
    print('[학습 시작]')
    print('root      :', os.path.abspath(ROOT_DIR))
    print('split root:', os.path.abspath(SPLIT_ROOT))
    print('liquid dir:', os.path.abspath(LIQUID_DIR))
    print('floor dir :', os.path.abspath(FLOOR_DIR))
    print('save dir  :', os.path.abspath(SAVE_DIR))
    print('=' * 90)

    encoder = LabelEncoder().fit(CLASSES)
    with open(os.path.join(SAVE_DIR, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

    liquid_files = collect_files(LIQUID_DIR)
    floor_files = collect_files(FLOOR_DIR)

    floor_to_liquid = defaultdict(list)
    floor_to_floor = defaultdict(list)

    for fp in liquid_files:
        name = os.path.basename(fp)
        floor_type = detect_floor_type(name)
        if floor_type:
            floor_to_liquid[floor_type].append(fp)

    for fp in floor_files:
        name = os.path.basename(fp)
        floor_type = detect_floor_type(name)
        if floor_type:
            floor_to_floor[floor_type].append(fp)

    for floor_type in FLOOR_KEYWORDS:
        liquid_list = floor_to_liquid[floor_type]
        floor_list = floor_to_floor[floor_type]

        if not liquid_list and not floor_list:
            print(f'[SKIP] {floor_type}: 데이터 없음')
            continue

        X_all, y_all = [], []

        for fp in liquid_list:
            base, signal, _changes = read_csv_recalc_change(fp)
            if base is None:
                print(f'[SKIP] liquid 읽기 실패: {os.path.basename(fp)}')
                continue

            signal = smooth_signal(remove_spikes(signal))
            max_start = len(signal) - WINDOW_SIZE
            if max_start < 0:
                continue

            for start in range(0, max_start + 1, LIQUID_STRIDE):
                win = create_window(signal, start, base, floor_type)
                if win is None:
                    continue
                for _ in range(LIQUID_BOOST):
                    aug = win + np.random.normal(0, random.uniform(0.01, 0.035), win.shape)
                    aug[:3] *= random.uniform(0.96, 1.04)
                    X_all.append(aug.astype(np.float32))
                    y_all.append('액체')

        for fp in floor_list:
            base, signal, _changes = read_csv_recalc_change(fp)
            if base is None:
                print(f'[SKIP] floor 읽기 실패: {os.path.basename(fp)}')
                continue

            signal = smooth_signal(remove_spikes(signal))
            max_start = len(signal) - WINDOW_SIZE
            if max_start < 0:
                continue

            for start in range(0, max_start + 1, FLOOR_STRIDE):
                win = create_window(signal, start, base, floor_type)
                if win is None:
                    continue
                for _ in range(FLOOR_BOOST):
                    aug = win + np.random.normal(0, random.uniform(0.005, 0.02), win.shape)
                    aug[:3] *= random.uniform(0.98, 1.02)
                    X_all.append(aug.astype(np.float32))
                    y_all.append('바닥')

        if len(X_all) == 0:
            print(f'[SKIP] {floor_type}: 학습 샘플 없음')
            continue

        X = np.array(X_all, dtype=np.float32).transpose(0, 2, 1)
        y = encoder.transform(y_all)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))

        classes_present = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, class_weights)}

        model = build_model()
        save_path = os.path.join(SAVE_DIR, f'model_{floor_type}.keras')
        print(f'저장 경로: {os.path.abspath(save_path)}')

        uniq, cnt = np.unique(y_all, return_counts=True)
        print('-' * 90)
        print(f'[TRAIN] {floor_type}')
        print('liquid files:', len(liquid_list))
        print('floor files :', len(floor_list))
        print('samples     :', len(X))
        print('label dist  :', {u: int(c) for u, c in zip(uniq, cnt)})

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ]

        model.fit(
            X, y_cat,
            validation_split=0.2,
            epochs=60,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

    print('\n전체 학습 종료')

if __name__ == '__main__':
    train_models()
