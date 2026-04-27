# -*- coding: utf-8 -*-
"""
3차 상태유지 실험용 학습 코드 v2 (이진 분류: 바닥/액체)

핵심
- 출력 라벨은 '바닥', '액체' 두 개만 사용
- manual _rise/_fall 파일은 액체로 학습
- manual _flat/_floor 파일은 바닥으로 학습
- 짧은 manual 파일도 WINDOW 길이로 보간해서 사용
- rise/fall 파일의 앞/뒤 평탄부는 자동으로 바닥 후보로 추가 시도
- 학습 로그를 SAVE_DIR에 txt로 저장
"""

import os
import glob
import csv
import pickle
import random
import re
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = r"C:\Users\MASL\Desktop\코드개선\3차 수동\3rd_수동"
SAVE_DIR = r"./kumoh_lstm_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW = 15
STEP_SCENARIO = 5
SMOOTH_K = 5
MIN_MANUAL_LEN = 4

CLASSES = ['바닥', '액체']

LIQUID_ALIASES = {
    'water': '물', 'coffee': '커피', 'cola': '콜라', 'milk': '우유',
    'mango': '망고', 'oil': '기름', 'matcha': '말차', 'tomato': '토마토', 'watermelon': '수박'
}
FLOOR_KEYWORDS = {
    '검대': ['검대'],
    '회대': ['회대'],
    '황대': ['황대'],
    '207회바': ['greyfloor', '회색바닥'],
    '흰책상': ['white', '하양', '흰', 'whitedesk'],
    '나타': ['나타'],
    '회타': ['회타'],
    '나무': ['나무', 'wood'],
}
REVERSE_DIRECTION_FLOORS = {'검대'}

FLOOR_THRESHOLDS = {
    '황대': {'grad_enter': 0.65, 'change_enter': 5.5, 'flat_std': 1.00, 'clear_change': 3.2},
    '회대': {'grad_enter': 0.75, 'change_enter': 6.0, 'flat_std': 1.10, 'clear_change': 3.8},
    '나타': {'grad_enter': 0.85, 'change_enter': 6.5, 'flat_std': 1.20, 'clear_change': 4.2},
    '회타': {'grad_enter': 0.85, 'change_enter': 6.5, 'flat_std': 1.20, 'clear_change': 4.2},
    '검대': {'grad_enter': 0.75, 'change_enter': 6.0, 'flat_std': 1.10, 'clear_change': 3.8},
    '그마': {'grad_enter': 0.75, 'change_enter': 6.0, 'flat_std': 1.10, 'clear_change': 3.8},
    '207회바': {'grad_enter': 0.75, 'change_enter': 6.0, 'flat_std': 1.10, 'clear_change': 3.8},
    '흰책상': {'grad_enter': 0.75, 'change_enter': 6.0, 'flat_std': 1.10, 'clear_change': 3.8},
    '나무': {'grad_enter': 0.75, 'change_enter': 6.0, 'flat_std': 1.10, 'clear_change': 3.8},
}

RAW_RE = re.compile(
    r"Time=(?P<time>[-+]?\d+(?:\.\d+)?)s\s*\|\s*"
    r"BaseRaw=(?P<base>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"CurrentRaw=(?P<current>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"Change=(?P<change>[-+]?\d+(?:\.\d+)?)"
)


def log(msg: str, fp=None):
    print(msg)
    if fp is not None:
        fp.write(msg + "\n")
        fp.flush()


def to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def smooth_signal(x, k=SMOOTH_K):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode='same')


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


def resample_to_len(x, target_len=WINDOW):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == target_len:
        return x
    if len(x) == 1:
        return np.repeat(x, target_len).astype(np.float32)
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst, src, x).astype(np.float32)


def detect_floor_type(filename: str):
    lower = filename.lower()
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None


def detect_manual_event_file(filename: str):
    lower = filename.lower()
    return ('_rise' in lower) or ('_fall' in lower)


def detect_manual_floor_file(filename: str):
    lower = filename.lower()
    return any(key in lower for key in ['_flat', '_floor', 'flat', 'floor'])


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

        if len(currents) < MIN_MANUAL_LEN:
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

        if len(currents) < MIN_MANUAL_LEN:
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
    if start + WINDOW > len(signal):
        return None
    sig = signal[start:start + WINDOW]
    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])

    base_eff = base if len(signal) < 20 else 0.7 * base + 0.3 * float(np.mean(signal[:20]))
    ctx_start = max(0, start - 5)
    ctx_end = min(len(signal), start + WINDOW + 5)
    ctx_mean = float(np.mean(signal[ctx_start:ctx_end]))
    direction_raw = base_eff - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - base_eff
    direction = np.clip(direction_raw / 300.0, -1.0, 1.0)
    direction_ch = np.full(WINDOW, direction, dtype=np.float32)

    return np.stack([norm, d1, d2, direction_ch]).astype(np.float32)


def stable_floor_window(sig_w, base, floor_type):
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    grad = np.gradient(sig_w)
    mean_change = float(np.mean(np.abs(sig_w - base)))
    max_grad = float(np.max(np.abs(grad)))
    flat_std = float(np.std(sig_w))
    return (
        mean_change <= params['clear_change'] and
        max_grad <= params['grad_enter'] and
        flat_std <= params['flat_std'] * 2.0
    )


def event_like_window(sig_w, base, floor_type):
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    grad = np.gradient(sig_w)
    abs_grad = np.abs(grad)
    max_grad = float(np.max(abs_grad))
    edge_ratio = float(np.mean(abs_grad >= params['grad_enter']))
    mean_change = float(np.mean(np.abs(sig_w - base)))
    return (
        mean_change >= params['change_enter'] and
        max_grad >= params['grad_enter'] and
        edge_ratio >= 0.25
    )


def add_manual_samples(signal, base, floor_type, label, X_all, y_all, fp_log, src_name):
    signal = smooth_signal(remove_spikes(signal))
    added = 0
    if len(signal) < WINDOW:
        win = create_window(resample_to_len(signal, WINDOW), 0, base, floor_type)
        if win is not None:
            X_all.append(win)
            y_all.append(label)
            added += 1
        return added

    step = 1 if label == '액체' else max(1, WINDOW // 3)
    for start in range(0, len(signal) - WINDOW + 1, step):
        sig_w = signal[start:start + WINDOW]
        win = create_window(signal, start, base, floor_type)
        if win is None:
            continue
        if label == '바닥' and not stable_floor_window(sig_w, base, floor_type):
            continue
        if label == '액체' and not event_like_window(sig_w, base, floor_type):
            # manual event 파일은 너무 약한 구간이면 건너뜀
            continue
        X_all.append(win)
        y_all.append(label)
        added += 1

    # 액체 파일에서 앞/뒤 평탄 구간 자동 추출
    if label == '액체' and len(signal) >= WINDOW:
        prefix = signal[:WINDOW]
        suffix = signal[-WINDOW:]
        for extra in [prefix, suffix]:
            if stable_floor_window(extra, base, floor_type):
                win = create_window(extra, 0, base, floor_type)
                if win is not None:
                    X_all.append(win)
                    y_all.append('바닥')
                    added += 1
    log(f"  [ADD] {src_name} | label={label} | +{added}", fp_log)
    return added


def build_model():
    inp = Input(shape=(WINDOW, 4))
    x = Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.20)(x)
    x = Bidirectional(LSTM(48, return_sequences=False))(x)
    x = Dropout(0.25)(x)
    out = Dense(2, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_models(data_dir=DATA_DIR, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"train_state_anchor_binary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_path, 'w', encoding='utf-8') as fp_log:
        log('=' * 90, fp_log)
        log('상태유지 v2 이진학습 시작', fp_log)
        log(f'[DB] {os.path.abspath(data_dir)}', fp_log)
        log(f'[SAVE] {os.path.abspath(save_dir)}', fp_log)

        encoder = LabelEncoder().fit(CLASSES)
        encoder_path = os.path.join(save_dir, 'encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        log(f'[encoder 저장] {os.path.abspath(encoder_path)}', fp_log)

        floor_files = {f: [] for f in FLOOR_KEYWORDS}
        all_csv = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        log(f'[CSV 개수] {len(all_csv)}', fp_log)
        for fp in all_csv:
            floor = detect_floor_type(os.path.basename(fp))
            if floor:
                floor_files[floor].append(fp)

        for floor, file_list in floor_files.items():
            if not file_list:
                log(f'[SKIP] {floor}: 파일 없음', fp_log)
                continue

            X_all, y_all = [], []
            log('-' * 90, fp_log)
            log(f'[FLOOR] {floor} | files={len(file_list)}', fp_log)

            for fp in file_list:
                name = os.path.basename(fp)
                base, signal, changes = read_csv_recalc_change(fp)
                if base is None:
                    log(f'[SKIP] 읽기 실패: {name}', fp_log)
                    continue

                is_event = detect_manual_event_file(name)
                is_floor = detect_manual_floor_file(name)
                signal = np.asarray(signal, dtype=np.float32)
                log(f'[LOAD] {name} | len={len(signal)} | base={base:.2f} | event={is_event} | floorfile={is_floor}', fp_log)

                if is_event:
                    add_manual_samples(signal, base, floor, '액체', X_all, y_all, fp_log, name)
                elif is_floor:
                    add_manual_samples(signal, base, floor, '바닥', X_all, y_all, fp_log, name)
                else:
                    # 일반 시나리오/기타 파일: 윈도우별 자동 라벨링
                    signal2 = smooth_signal(remove_spikes(signal))
                    if len(signal2) < WINDOW:
                        continue
                    for start in range(0, len(signal2) - WINDOW + 1, STEP_SCENARIO):
                        sig_w = signal2[start:start + WINDOW]
                        win = create_window(signal2, start, base, floor)
                        if win is None:
                            continue
                        if event_like_window(sig_w, base, floor):
                            X_all.append(win)
                            y_all.append('액체')
                        elif stable_floor_window(sig_w, base, floor):
                            X_all.append(win)
                            y_all.append('바닥')

            if len(X_all) == 0:
                log(f'[SKIP] {floor}: 학습 데이터가 없습니다.', fp_log)
                continue

            # augment
            X_aug, y_aug = [], []
            for x, y in zip(X_all, y_all):
                aug_count = 3 if y == '바닥' else 2
                for _ in range(aug_count):
                    aug = x + np.random.normal(0, random.uniform(0.005, 0.03), x.shape)
                    aug[:3] *= random.uniform(0.97, 1.03)
                    X_aug.append(aug.astype(np.float32))
                    y_aug.append(y)

            X = np.array(X_aug, dtype=np.float32).transpose(0, 2, 1)
            y = encoder.transform(y_aug)
            y_cat = tf.keras.utils.to_categorical(y, num_classes=2)

            classes_present = np.unique(y)
            class_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y)
            class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, class_weights)}

            uniq, cnt = np.unique(y_aug, return_counts=True)
            dist = {str(u): int(c) for u, c in zip(uniq, cnt)}
            log(f'[TRAIN] {floor} | samples={len(X)} | dist={dist}', fp_log)

            model = build_model()
            save_path = os.path.join(save_dir, f'model_{floor}.keras')
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
                verbose=1,
            )
            log(f'[모델 저장] {os.path.abspath(save_path)}', fp_log)

        log('=' * 90, fp_log)
        log('학습 완료', fp_log)
        log(f'[학습 로그] {os.path.abspath(log_path)}', fp_log)


if __name__ == '__main__':
    train_models()
