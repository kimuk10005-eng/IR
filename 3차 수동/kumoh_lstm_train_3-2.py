# -*- coding: utf-8 -*-
"""
기존 2-4 구조 유지 + 로컬 베이스라인 + Moving Average 반영 학습 코드

핵심
- 기존 4채널 유지: norm_sig, deriv1, deriv2, direction_channel
- 기존 WINDOW 기반 유지
- 로컬 베이스라인(local baseline) 추가
- moving average change / moving average gradient 반영
- 평탄 구간은 바닥으로 더 강하게
- 이벤트성 rise/fall 윈도우를 액체로 더 우선 학습
"""

import os
import glob
import csv
import pickle
import random
import re

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===========================
# 설정
# ===========================
DATA_DIR = r"C:\Users\MASL\Desktop\데이터 크롭\예시\trimmed_manual_split"
SAVE_DIR = "./kumoh_lstm_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 15
SMOOTH_K = 5
MOVING_AVG_K = 7
LOCAL_BASELINE_MARGIN = 12
LOCAL_BASELINE_MIN_POINTS = 5

CHANGE_THRESHOLD = 5.5
LOW_CHANGE_THRESHOLD = 4.5
PLATEAU_MEAN_THRESHOLD = 7.0
PLATEAU_MEDIAN_THRESHOLD = 6.0
SIGN_CONSISTENCY_THRESHOLD = 0.88
NOISE_RATIO_THRESHOLD = 0.82
SPIKE_RATIO_THRESHOLD = 2.20

EDGE_GRAD_THRESHOLD = 0.85
EDGE_RATIO_THRESHOLD = 0.28
FLAT_STD_THRESHOLD = 0.75
FLAT_GRAD_THRESHOLD = 0.45
PLATEAU_KEEP_PROB = 0.15

LIQUIDS = ['바닥', '물', '말차', '커피', '콜라', '토마토', '우유', '망고', '기름', '수박']

LIQUID_ALIASES = {
    'water': '물',
    'coffee': '커피',
    'cola': '콜라',
    'milk': '우유',
    'mango': '망고',
    'oil': '기름',
    'matcha': '말차',
    'tomato': '토마토',
    'watermelon': '수박',
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
        return x
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(np.asarray(x, dtype=np.float32), kernel, mode='same')

def moving_average(x, k=MOVING_AVG_K):
    if len(x) == 0:
        return x
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

def compute_noise_metrics(diff_w, smooth_w):
    mean_abs_diff = float(np.mean(np.abs(diff_w))) + 1e-6
    noise_ratio = float(np.std(diff_w) / mean_abs_diff)
    spike_ratio = float((np.max(smooth_w) - np.min(smooth_w)) / mean_abs_diff)
    return noise_ratio, spike_ratio

def detect_liquid_from_filename(filename: str):
    lower = filename.lower()
    for key, kor in LIQUID_ALIASES.items():
        if key in lower:
            return kor
    for kor in LIQUIDS:
        if kor != '바닥' and kor in filename:
            return kor
    return None

def detect_floor_type(filename: str):
    lower = filename.lower()
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None

def detect_manual_event_file(filename: str):
    lower = filename.lower()
    return ('_rise' in lower) or ('_fall' in lower)

def read_csv_recalc_change(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    def parse_new_format(lines):
        baseline_values = []
        currents = []
        row_bases = []
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
        baseline_values = []
        currents = []
        row_bases = []
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
                    key = parts[i]
                    val = parts[i + 1]
                    if key == 'BaseRaw':
                        base_raw = to_float(val)
                    elif key == 'CurrentRaw':
                        current_raw = to_float(val)
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

            base, currents, changes = parse_new_format(lines)
            if base is not None:
                return base, currents, changes

            base, currents, changes = parse_old_format(lines)
            if base is not None:
                return base, currents, changes
        except Exception:
            continue

    return None, None, None

def get_floor_ref_params(floor_type):
    if floor_type in {'나타', '회타'}:
        return 0.020, 0.65, 5.0, 0.68, 1.85
    if floor_type == '황대':
        return 0.028, 0.75, 6.0, 0.82, 2.20
    return 0.026, 0.75, 6.0, 0.78, 2.05

def build_recent_floor_reference(signal, baseline, floor_type):
    alpha, ref_grad, ref_band, _, _ = get_floor_ref_params(floor_type)
    sig = smooth_signal(remove_spikes(signal))
    grad = np.gradient(sig)
    ref = np.empty_like(sig)
    ref[0] = baseline
    for i in range(1, len(sig)):
        stable_grad = abs(grad[i]) < ref_grad
        close_to_ref = abs(sig[i] - ref[i - 1]) < ref_band
        if stable_grad and close_to_ref:
            ref[i] = (1.0 - alpha) * ref[i - 1] + alpha * sig[i]
        else:
            ref[i] = ref[i - 1]
    return sig, grad, ref

def compute_local_baseline(signal, start, window_size=WINDOW_SIZE):
    left = max(0, start - LOCAL_BASELINE_MARGIN)
    right = max(0, start - 1)
    if right - left + 1 >= LOCAL_BASELINE_MIN_POINTS:
        return float(np.mean(signal[left:right + 1]))
    left2 = start + window_size
    right2 = min(len(signal) - 1, start + window_size + LOCAL_BASELINE_MARGIN - 1)
    if right2 - left2 + 1 >= LOCAL_BASELINE_MIN_POINTS:
        return float(np.mean(signal[left2:right2 + 1]))
    return None

def create_window(signal, start, base, floor_type, local_base=None):
    if start + WINDOW_SIZE > len(signal):
        return None

    sig = signal[start:start + WINDOW_SIZE]
    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])

    if local_base is None:
        local_base = base if len(signal) < 20 else 0.7 * base + 0.3 * np.mean(signal[:20])

    ctx_start = max(0, start - 5)
    ctx_end = min(len(signal), start + WINDOW_SIZE + 5)
    ctx_mean = np.mean(signal[ctx_start:ctx_end])
    direction_raw = local_base - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - local_base
    direction = np.clip(direction_raw / 300.0, -1.0, 1.0)
    direction_ch = np.full(WINDOW_SIZE, direction)

    return np.stack([norm, d1, d2, direction_ch])

def classify_window_for_training(sig_w, chg_w, diff_w, grad_w, ma_change_w, ma_grad_w,
                                 floor_type, liquid, is_manual_event=False):
    max_change = float(np.max(chg_w))
    mean_floor_diff = float(np.mean(np.abs(diff_w)))
    median_floor_diff = float(np.abs(np.median(diff_w)))
    sign_consistency = float(max(np.mean(diff_w >= 0), np.mean(diff_w <= 0)))
    noise_ratio, spike_ratio = compute_noise_metrics(diff_w, sig_w)

    abs_grad = np.abs(grad_w)
    edge_ratio = float(np.mean(abs_grad >= EDGE_GRAD_THRESHOLD))
    max_gradient = float(np.max(abs_grad))
    flat_std = float(np.std(sig_w))
    ma_change = float(np.mean(ma_change_w))
    ma_gradient = float(np.mean(ma_grad_w))

    _, _, _, floor_noise_th, floor_spike_th = get_floor_ref_params(floor_type)

    strong_liquid = max_change >= CHANGE_THRESHOLD
    plateau_liquid = (
        max_change >= LOW_CHANGE_THRESHOLD
        and mean_floor_diff >= PLATEAU_MEAN_THRESHOLD
        and median_floor_diff >= PLATEAU_MEDIAN_THRESHOLD
        and sign_consistency >= SIGN_CONSISTENCY_THRESHOLD
        and noise_ratio <= min(NOISE_RATIO_THRESHOLD, floor_noise_th)
        and spike_ratio <= min(SPIKE_RATIO_THRESHOLD, floor_spike_th)
    )
    clear_edge_event = (
        max_gradient >= EDGE_GRAD_THRESHOLD
        and edge_ratio >= EDGE_RATIO_THRESHOLD
        and ma_gradient >= EDGE_GRAD_THRESHOLD * 0.75
        and ma_change >= LOW_CHANGE_THRESHOLD * 0.85
    )
    clear_flat_floor = (
        flat_std <= FLAT_STD_THRESHOLD
        and max_gradient <= FLAT_GRAD_THRESHOLD
        and ma_gradient <= FLAT_GRAD_THRESHOLD * 0.95
        and ma_change < LOW_CHANGE_THRESHOLD * 0.80
        and mean_floor_diff < PLATEAU_MEAN_THRESHOLD * 0.75
    )

    if is_manual_event:
        if clear_edge_event and max_change >= LOW_CHANGE_THRESHOLD:
            return liquid
        if plateau_liquid and ma_gradient >= EDGE_GRAD_THRESHOLD * 0.60 and random.random() < PLATEAU_KEEP_PROB:
            return liquid
        return None

    if clear_flat_floor:
        return '바닥'
    if strong_liquid and clear_edge_event:
        return liquid
    if plateau_liquid and ma_gradient >= EDGE_GRAD_THRESHOLD * 0.60 and random.random() < PLATEAU_KEEP_PROB:
        return liquid
    if (not strong_liquid) and (not plateau_liquid):
        return '바닥'
    return None

def build_model():
    inp = Input(shape=(WINDOW_SIZE, 4))
    x = Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    out = Dense(len(LIQUIDS), activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models(data_dir=DATA_DIR, save_dir=SAVE_DIR):
    encoder = LabelEncoder().fit(LIQUIDS)
    with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

    floor_files = {f: [] for f in FLOOR_KEYWORDS}
    for fp in glob.glob(os.path.join(data_dir, '*.csv')):
        floor = detect_floor_type(os.path.basename(fp))
        if floor:
            floor_files[floor].append(fp)

    for floor, file_list in floor_files.items():
        if not file_list:
            print(f'[SKIP] {floor}: 파일 없음')
            continue

        X_all, y_all = [], []

        for fp in file_list:
            name = os.path.basename(fp)
            liquid = detect_liquid_from_filename(name)
            is_manual_event = detect_manual_event_file(name)

            if liquid is None and is_manual_event:
                continue

            base, signal, changes = read_csv_recalc_change(fp)
            if base is None:
                print(f'[SKIP] 읽기 실패: {name}')
                continue

            smooth_sig, grad_sig, floor_ref = build_recent_floor_reference(signal, base, floor)
            ma_changes = moving_average(changes, MOVING_AVG_K)
            ma_grads = moving_average(np.abs(grad_sig), MOVING_AVG_K)

            max_start = len(signal) - WINDOW_SIZE
            if max_start <= 0:
                continue

            step = 1 if is_manual_event else max(1, max_start // 25)

            for start in range(0, max_start + 1, step):
                local_base = compute_local_baseline(smooth_sig, start, WINDOW_SIZE)
                if local_base is None:
                    local_base = base

                win = create_window(smooth_sig, start, base, floor, local_base=local_base)
                if win is None:
                    continue

                sig_w = smooth_sig[start:start + WINDOW_SIZE]
                chg_w = np.abs(sig_w - local_base).astype(np.float32)
                ref_w = floor_ref[start:start + WINDOW_SIZE]
                diff_w = sig_w - ref_w
                grad_w = grad_sig[start:start + WINDOW_SIZE]
                ma_change_w = ma_changes[start:start + WINDOW_SIZE]
                ma_grad_w = ma_grads[start:start + WINDOW_SIZE]

                label = classify_window_for_training(
                    sig_w=sig_w,
                    chg_w=chg_w,
                    diff_w=diff_w,
                    grad_w=grad_w,
                    ma_change_w=ma_change_w,
                    ma_grad_w=ma_grad_w,
                    floor_type=floor,
                    liquid=liquid if liquid is not None else '바닥',
                    is_manual_event=is_manual_event
                )

                if label is None:
                    continue

                aug_count = 2 if label == '바닥' else 3
                for _ in range(aug_count):
                    aug = win + np.random.normal(0, random.uniform(0.01, 0.04), win.shape)
                    aug[:3] *= random.uniform(0.95, 1.05)
                    X_all.append(aug.astype(np.float32))
                    y_all.append(label)

        if len(X_all) == 0:
            print(f'[SKIP] {floor}: 학습 데이터가 없습니다.')
            continue

        X = np.array(X_all, dtype=np.float32).transpose(0, 2, 1)
        y = encoder.transform(y_all)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(LIQUIDS))

        classes_present = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, class_weights)}

        model = build_model()
        save_path = os.path.join(save_dir, f'model_{floor}.keras')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ]

        print('=' * 90)
        print(f'[TRAIN] 바닥={floor} | 파일수={len(file_list)} | 샘플수={len(X)}')
        uniq, cnt = np.unique(y_all, return_counts=True)
        print('라벨분포:', {u: int(c) for u, c in zip(uniq, cnt)})

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
