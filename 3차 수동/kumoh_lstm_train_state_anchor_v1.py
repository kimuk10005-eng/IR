# -*- coding: utf-8 -*-
"""
3차 상태유지 실험용 학습 코드 v1
- 4채널 유지: norm, deriv1, deriv2, direction_channel
- manual split(_rise/_fall)은 액체 이벤트로 학습
- _flat / _floor / floor / flat 파일은 바닥으로 학습
- 일반 시나리오 파일이 있으면 바닥/액체 윈도우를 추가로 추출
- 로그를 모델 저장 폴더에 txt로 저장
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

plt_disabled = True

DATA_DIR = r"C:\Users\MASL\Desktop\코드개선\3차 수동\3rd_수동"
SAVE_DIR = r"./kumoh_lstm_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW = 15
STEP_SCENARIO = 5
SMOOTH_K = 5

LIQUIDS = ['바닥', '물', '말차', '커피', '콜라', '토마토', '우유', '망고', '기름', '수박']
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
    '황대': {'grad_enter': 0.65, 'change_enter': 5.5, 'hold_offset': 6.0, 'flat_std': 1.00},
    '회대': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'flat_std': 1.10},
    '나타': {'grad_enter': 0.85, 'change_enter': 6.5, 'hold_offset': 7.5, 'flat_std': 1.20},
    '회타': {'grad_enter': 0.85, 'change_enter': 6.5, 'hold_offset': 7.5, 'flat_std': 1.20},
    '검대': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'flat_std': 1.10},
    '그마': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'flat_std': 1.10},
    '207회바': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'flat_std': 1.10},
    '흰책상': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'flat_std': 1.10},
    '나무': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'flat_std': 1.10},
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


def is_manual_event_file(filename: str):
    lower = filename.lower()
    return ('_rise' in lower) or ('_fall' in lower)


def is_manual_floor_file(filename: str):
    lower = filename.lower()
    return any(tag in lower for tag in ['_flat', '_floor', ' flat', ' floor'])


def resample_to_len(x, target_len=WINDOW):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == target_len:
        return x
    if len(x) == 1:
        return np.full((target_len,), float(x[0]), dtype=np.float32)
    old_idx = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def read_csv_recalc_change(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    def parse_new(lines):
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
        if len(currents) < 2:
            return None, None, None
        if baseline_values:
            base = float(np.mean(baseline_values))
        elif row_bases:
            base = float(np.mean(row_bases))
        else:
            return None, None, None
        currents = np.array(currents, dtype=np.float32)
        changes = np.abs(currents - base).astype(np.float32)
        return base, currents, changes

    def parse_old(lines):
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
                base_raw, current_raw = None, None
                for i in range(len(parts) - 1):
                    if parts[i] == 'BaseRaw':
                        base_raw = to_float(parts[i + 1])
                    elif parts[i] == 'CurrentRaw':
                        current_raw = to_float(parts[i + 1])
                if current_raw is not None:
                    currents.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)
        if len(currents) < 2:
            return None, None, None
        if final_base is not None:
            base = float(final_base)
        elif baseline_values:
            base = float(np.mean(baseline_values))
        elif row_bases:
            base = float(np.mean(row_bases))
        else:
            return None, None, None
        currents = np.array(currents, dtype=np.float32)
        changes = np.abs(currents - base).astype(np.float32)
        return base, currents, changes

    for enc in encodings:
        try:
            with open(fp, 'r', encoding=enc, errors='ignore', newline='') as f:
                lines = f.readlines()
            r = parse_new(lines)
            if r[0] is not None:
                return r
            r = parse_old(lines)
            if r[0] is not None:
                return r
        except Exception:
            continue
    return None, None, None


def build_recent_floor_reference(signal, baseline):
    sig = smooth_signal(remove_spikes(signal))
    grad = np.gradient(sig)
    ref = np.empty_like(sig)
    ref[0] = baseline
    alpha = 0.03
    for i in range(1, len(sig)):
        if abs(grad[i]) < 0.8 and abs(sig[i] - ref[i - 1]) < 6.0:
            ref[i] = (1.0 - alpha) * ref[i - 1] + alpha * sig[i]
        else:
            ref[i] = ref[i - 1]
    return sig, grad, ref


def create_window(signal, start, baseline, floor_type, anchor_base=None):
    if start + WINDOW > len(signal):
        return None
    sig = signal[start:start + WINDOW]
    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])
    if anchor_base is None:
        anchor_base = baseline
    ctx_start = max(0, start - 5)
    ctx_end = min(len(signal), start + WINDOW + 5)
    ctx_mean = float(np.mean(signal[ctx_start:ctx_end]))
    direction_raw = anchor_base - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - anchor_base
    direction = np.clip(direction_raw / 300.0, -1.0, 1.0)
    direction_ch = np.full(WINDOW, direction, dtype=np.float32)
    return np.stack([norm, d1, d2, direction_ch], axis=0)


def classify_scenario_window(sig_w, grad_w, baseline, floor_type):
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    mean_abs_offset = float(np.mean(np.abs(sig_w - baseline)))
    max_abs_grad = float(np.max(np.abs(grad_w)))
    std = float(np.std(sig_w))
    edge_ratio = float(np.mean(np.abs(grad_w) >= params['grad_enter']))

    clear_floor = (
        mean_abs_offset < params['hold_offset'] * 0.45
        and max_abs_grad < params['grad_enter'] * 0.85
        and std < params['flat_std']
    )
    clear_event = (
        mean_abs_offset >= params['change_enter']
        and max_abs_grad >= params['grad_enter']
        and edge_ratio >= 0.20
    )
    if clear_floor:
        return '바닥'
    if clear_event:
        return '액체'
    return None


def build_model(num_classes: int):
    inp = Input(shape=(WINDOW, 4))
    x = Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_models(data_dir=DATA_DIR, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f'train_state_anchor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(log_path, 'w', encoding='utf-8') as log_fp:
        log('=' * 90, log_fp)
        log('상태유지 실험용 학습 시작', log_fp)
        log(f'[DB 경로] {os.path.abspath(data_dir)}', log_fp)
        log(f'[모델 저장 경로] {os.path.abspath(save_dir)}', log_fp)

        encoder = LabelEncoder().fit(LIQUIDS)
        encoder_path = os.path.join(save_dir, 'encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        log(f'[encoder 저장] {os.path.abspath(encoder_path)}', log_fp)

        floor_files = {f: [] for f in FLOOR_KEYWORDS}
        for fp in glob.glob(os.path.join(data_dir, '*.csv')):
            floor = detect_floor_type(os.path.basename(fp))
            if floor:
                floor_files[floor].append(fp)
        log(f'[검색된 CSV 수] {sum(len(v) for v in floor_files.values())}', log_fp)

        for floor, file_list in floor_files.items():
            if not file_list:
                log(f'[SKIP] {floor}: 파일 없음', log_fp)
                continue

            X_all, y_all = [], []
            label_counts = {}

            for fp in sorted(file_list):
                name = os.path.basename(fp)
                liquid = detect_liquid_from_filename(name)
                manual_event = is_manual_event_file(name)
                manual_floor = is_manual_floor_file(name)

                base, signal, _changes = read_csv_recalc_change(fp)
                if base is None:
                    log(f'[SKIP] 읽기 실패: {name}', log_fp)
                    continue

                signal = smooth_signal(remove_spikes(signal))
                original_len = len(signal)
                added = 0
                mode = 'scenario'

                if manual_event or manual_floor:
                    mode = 'manual-event' if manual_event else 'manual-floor'
                    rs = resample_to_len(signal, WINDOW)
                    anchor = float(base)
                    win = create_window(rs, 0, base, floor, anchor)
                    if win is not None:
                        label = liquid if manual_event and liquid is not None else '바닥'
                        aug_n = 4 if label != '바닥' else 3
                        for _ in range(aug_n):
                            aug = win + np.random.normal(0, random.uniform(0.005, 0.03), win.shape)
                            X_all.append(aug.astype(np.float32))
                            y_all.append(label)
                            added += 1
                        label_counts[label] = label_counts.get(label, 0) + added
                    log(f'[LOAD] {name} | mode={mode} | len={original_len}->{WINDOW} | base={base:.2f} | label={label if win is not None else "skip"} | +{added}', log_fp)
                    continue

                # 일반 시나리오 파일
                if liquid is None:
                    liquid = '바닥'
                sig_smooth, grad_sig, _ref = build_recent_floor_reference(signal, base)
                max_start = len(sig_smooth) - WINDOW
                if max_start < 0:
                    log(f'[SKIP] 너무 짧음: {name}', log_fp)
                    continue

                for start in range(0, max_start + 1, STEP_SCENARIO):
                    sig_w = sig_smooth[start:start + WINDOW]
                    grad_w = grad_sig[start:start + WINDOW]
                    coarse = classify_scenario_window(sig_w, grad_w, base, floor)
                    if coarse is None:
                        continue
                    if coarse == '바닥':
                        label = '바닥'
                    else:
                        label = liquid
                    win = create_window(sig_smooth, start, base, floor, base)
                    if win is None:
                        continue
                    aug_n = 2 if label == '바닥' else 2
                    for _ in range(aug_n):
                        aug = win + np.random.normal(0, random.uniform(0.005, 0.03), win.shape)
                        X_all.append(aug.astype(np.float32))
                        y_all.append(label)
                        added += 1
                label_counts = {k: label_counts.get(k, 0) for k in label_counts}
                log(f'[LOAD] {name} | mode={mode} | len={original_len} | base={base:.2f} | +{added}', log_fp)

            if len(X_all) == 0:
                log(f'[SKIP] {floor}: 학습 데이터가 없습니다.', log_fp)
                continue

            X = np.array(X_all, dtype=np.float32).transpose(0, 2, 1)
            y = encoder.transform(y_all)
            y_cat = tf.keras.utils.to_categorical(y, num_classes=len(LIQUIDS))

            classes_present = np.unique(y)
            class_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y)
            class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, class_weights)}

            model = build_model(len(LIQUIDS))
            save_path = os.path.join(save_dir, f'model_{floor}.keras')
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
                ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
            ]

            uniq, cnt = np.unique(y_all, return_counts=True)
            dist = {str(u): int(c) for u, c in zip(uniq, cnt)}
            log('=' * 90, log_fp)
            log(f'[TRAIN] 바닥={floor} | 파일수={len(file_list)} | 샘플수={len(X)}', log_fp)
            log(f'[라벨분포] {dist}', log_fp)
            log(f'[모델 저장] {os.path.abspath(save_path)}', log_fp)

            model.fit(
                X, y_cat,
                validation_split=0.2,
                epochs=60,
                batch_size=32,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1,
            )

        log('=' * 90, log_fp)
        log('학습 완료', log_fp)
        log(f'[학습 로그 저장] {os.path.abspath(log_path)}', log_fp)


if __name__ == '__main__':
    train_models()
