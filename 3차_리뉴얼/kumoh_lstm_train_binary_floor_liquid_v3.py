# -*- coding: utf-8 -*-
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
    r"Time\s*=?\s*(?P<time>[-+]?\d+(?:\.\d+)?)s?\s*[\|,]?\s*"
    r".*?BaseRaw\s*=?\s*(?P<base>[-+]?\d+(?:\.\d+)?)"
    r".*?CurrentRaw\s*=?\s*(?P<current>[-+]?\d+(?:\.\d+)?)"
    r".*?Change\s*=?\s*(?P<change>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE
)
GRAPH_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def ask_path(prompt, default=""):
    msg = f"{prompt}"
    if default:
        msg += f" [{default}]"
    msg += " ▶ "
    v = input(msg).strip().strip('"')
    return v if v else default

def to_float(x):
    try:
        s = str(x).strip()
        if s == '' or s.lower() == 'nan':
            return None
        return float(s)
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
        if abs(float(sig[i]) - local_med) > k * local_std and abs(float(sig[i]) - float(sig[i - 1])) > 1.2 * local_std and abs(float(sig[i]) - float(sig[i + 1])) > 1.2 * local_std:
            sig[i] = np.float32(0.5 * (sig[i - 1] + sig[i + 1]))
    return sig

def detect_floor_type(filename: str):
    lower = filename.lower()
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None

def extract_graph_series(graph_values):
    vals = []
    for g in graph_values:
        if g in (None, ''):
            continue
        nums = GRAPH_NUM_RE.findall(str(g))
        vals.extend([float(n) for n in nums])
    return vals

def read_csv_recalc_change(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            with open(fp, 'r', encoding=enc, errors='ignore', newline='') as f:
                lines = f.read().splitlines()
            reader = csv.DictReader(lines)
            rows = list(reader)
            fields = [str(x).strip() for x in (reader.fieldnames or [])]
            if 'record_type' not in fields:
                continue

            baseline_values, row_bases, currents, graph_values = [], [], [], []
            for row in rows:
                record_type = str(row.get('record_type', '')).strip().lower()
                baseline_raw = to_float(row.get('baseline_raw', ''))
                base_raw = to_float(row.get('base_raw', ''))
                current_raw = to_float(row.get('current_raw', ''))
                raw_data = str(row.get('raw_data', '')).strip()
                graph_values.append(row.get('graph', ''))

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
                g = extract_graph_series(graph_values)
                if len(g) >= WINDOW_SIZE:
                    currents = g

            if len(currents) < WINDOW_SIZE:
                return None, None, None, "data/current_raw 부족"

            if baseline_values:
                base = float(np.mean(baseline_values))
            elif row_bases:
                base = float(np.mean(row_bases))
            else:
                base = float(np.median(currents[:min(10, len(currents))]))

            currents = np.array(currents, dtype=np.float32)
            changes = np.abs(currents - base).astype(np.float32)
            return base, currents, changes, None
        except Exception:
            continue
    return None, None, None, "encoding/format 실패"

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
    return sorted(glob.glob(os.path.join(folder, '*.csv'))) if os.path.isdir(folder) else []

def train_models(root_dir, split_root, liquid_dir, floor_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print("\n[학습 시작]")
    print("split root:", split_root)
    print("liquid dir:", liquid_dir)
    print("floor dir :", floor_dir)
    print("save dir  :", save_dir)

    encoder = LabelEncoder().fit(CLASSES)
    with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

    liquid_files = collect_files(liquid_dir)
    floor_files = collect_files(floor_dir)

    # preview
    print(f"liquid 파일 수: {len(liquid_files)} / floor 파일 수: {len(floor_files)}")
    for fp in liquid_files[:3]:
        b, s, c, err = read_csv_recalc_change(fp)
        print("[liquid preview]", os.path.basename(fp), "len=", None if s is None else len(s), "err=", err)
    for fp in floor_files[:3]:
        b, s, c, err = read_csv_recalc_change(fp)
        print("[floor preview ]", os.path.basename(fp), "len=", None if s is None else len(s), "err=", err)

    floor_to_liquid = defaultdict(list)
    floor_to_floor = defaultdict(list)

    for fp in liquid_files:
        floor_type = detect_floor_type(os.path.basename(fp))
        if floor_type:
            floor_to_liquid[floor_type].append(fp)
    for fp in floor_files:
        floor_type = detect_floor_type(os.path.basename(fp))
        if floor_type:
            floor_to_floor[floor_type].append(fp)

    for floor_type in FLOOR_KEYWORDS:
        liquid_list = floor_to_liquid[floor_type]
        floor_list = floor_to_floor[floor_type]
        if not liquid_list and not floor_list:
            print(f"[SKIP] {floor_type}: 데이터 없음")
            continue

        X_all, y_all = [], []

        for fp in liquid_list:
            base, signal, _changes, err = read_csv_recalc_change(fp)
            if base is None:
                print(f"[SKIP] liquid 읽기 실패: {os.path.basename(fp)} | 이유: {err}")
                continue
            signal = smooth_signal(remove_spikes(signal))
            for start in range(0, len(signal) - WINDOW_SIZE + 1, LIQUID_STRIDE):
                win = create_window(signal, start, base, floor_type)
                if win is None:
                    continue
                for _ in range(LIQUID_BOOST):
                    aug = win + np.random.normal(0, random.uniform(0.01, 0.035), win.shape)
                    aug[:3] *= random.uniform(0.96, 1.04)
                    X_all.append(aug.astype(np.float32))
                    y_all.append('액체')

        for fp in floor_list:
            base, signal, _changes, err = read_csv_recalc_change(fp)
            if base is None:
                print(f"[SKIP] floor 읽기 실패: {os.path.basename(fp)} | 이유: {err}")
                continue
            signal = smooth_signal(remove_spikes(signal))
            for start in range(0, len(signal) - WINDOW_SIZE + 1, FLOOR_STRIDE):
                win = create_window(signal, start, base, floor_type)
                if win is None:
                    continue
                for _ in range(FLOOR_BOOST):
                    aug = win + np.random.normal(0, random.uniform(0.005, 0.02), win.shape)
                    aug[:3] *= random.uniform(0.98, 1.02)
                    X_all.append(aug.astype(np.float32))
                    y_all.append('바닥')

        if not X_all:
            print(f"[SKIP] {floor_type}: 학습 샘플 없음")
            continue

        X = np.array(X_all, dtype=np.float32).transpose(0, 2, 1)
        y = encoder.transform(y_all)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))

        classes_present = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, class_weights)}

        model = build_model()
        save_path = os.path.join(save_dir, f'model_{floor_type}.keras')

        uniq, cnt = np.unique(y_all, return_counts=True)
        print('-' * 90)
        print(f'[TRAIN] {floor_type}')
        print('liquid files:', len(liquid_list))
        print('floor files :', len(floor_list))
        print('samples     :', len(X))
        print('label dist  :', {u: int(c) for u, c in zip(uniq, cnt)})

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ]

        model.fit(
            X, y_cat,
            validation_split=0.2,
            epochs=40,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

    print('\n전체 학습 종료')

if __name__ == '__main__':
    default_root = r"C:\Users\MASL\Desktop\3차리뉴얼db"
    root_dir = ask_path("루트 폴더", default_root)
    split_root = ask_path("split 폴더", os.path.join(root_dir, "trimmed_manual_split"))
    liquid_dir = ask_path("liquid 폴더", os.path.join(split_root, "liquid"))
    floor_dir = ask_path("floor 폴더", os.path.join(split_root, "floor"))
    save_dir = ask_path("모델 저장 폴더", os.path.join(root_dir, "kumoh_binary_model_save"))
    train_models(root_dir, split_root, liquid_dir, floor_dir, save_dir)
