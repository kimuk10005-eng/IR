# -*- coding: utf-8 -*-
"""
2-4 구조 기반 이진 학습 코드 v1
- 출력 라벨: 바닥 / 액체
- 4채널 유지: norm, deriv1, deriv2, direction_channel
- 수동 rise/fall 파일 + 전체 시나리오 파일 둘 다 학습 가능
- 애매한 plateau 전체를 액체로 길게 학습하지 않고, 이벤트와 붙은 plateau만 제한적으로 사용
- 학습 로그를 SAVE_DIR에 저장
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

# =========================
# 경로 설정
# =========================
DATA_DIR = r"C:\Users\MASL\Desktop\ir data(get)\260325주 학습"
MANUAL_DIR = r"C:\Users\MASL\Desktop\코드개선\3차 수동\trimmed_manual_split"
SAVE_DIR = r"./kumoh_lstm_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 공통 설정
# =========================
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
}

REVERSE_DIRECTION_FLOORS = {'검대'}

FLOOR_THRESHOLDS = {
    '나무': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.88, 'alpha': 0.030, 'ref_grad': 0.80, 'ref_band': 6.5, 'noise_ratio': 0.90, 'spike_ratio': 2.40},
    '황대': {'gradient': 0.6, 'change_low': 5.0, 'change_strong': 15.0, 'plateau_mean': 7.0, 'plateau_med': 6.0, 'clear_mean': 2.3, 'sign_keep': 0.88, 'alpha': 0.028, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.82, 'spike_ratio': 2.20},
    '회대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05},
    '검대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05},
    '그마': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05},
    '207회바': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05},
    '흰책상': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05},
    '나타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'plateau_mean': 10.0, 'plateau_med': 8.0, 'clear_mean': 3.0, 'sign_keep': 0.90, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0, 'noise_ratio': 0.68, 'spike_ratio': 1.85},
    '회타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'plateau_mean': 10.0, 'plateau_med': 8.0, 'clear_mean': 3.0, 'sign_keep': 0.90, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0, 'noise_ratio': 0.68, 'spike_ratio': 1.85},
}

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


def detect_floor_type(filename: str):
    lower = filename.lower()
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None


def detect_manual_mode(filename: str):
    lower = filename.lower()
    if '_rise' in lower or '_fall' in lower:
        return 'liquid'
    if '_flat' in lower or '_floor' in lower or 'flat' in lower:
        return 'floor'
    return 'scenario'


def resize_signal(sig, target_len=WINDOW):
    sig = np.asarray(sig, dtype=np.float32)
    if len(sig) == target_len:
        return sig
    x_old = np.linspace(0, 1, len(sig))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, sig).astype(np.float32)


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


def build_recent_floor_reference(signal, baseline, floor_type):
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    sig = smooth_signal(remove_spikes(signal))
    grad = np.gradient(sig)
    ref = np.empty_like(sig)
    ref[0] = baseline
    alpha = params.get('alpha', 0.03)
    ref_grad = params.get('ref_grad', 0.8)
    ref_band = params.get('ref_band', 6.5)
    for i in range(1, len(sig)):
        stable_grad = abs(grad[i]) < ref_grad
        close_to_ref = abs(sig[i] - ref[i - 1]) < ref_band
        if stable_grad and close_to_ref:
            ref[i] = (1.0 - alpha) * ref[i - 1] + alpha * sig[i]
        else:
            ref[i] = ref[i - 1]
    return sig, grad, ref


def build_window(sig, base, floor_type, full_signal, start_idx):
    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])

    base_eff = base if len(full_signal) < 20 else 0.7 * base + 0.3 * np.mean(full_signal[:20])
    ctx_start = max(0, start_idx - 5)
    ctx_end = min(len(full_signal), start_idx + WINDOW + 5)
    ctx_mean = np.mean(full_signal[ctx_start:ctx_end])
    direction_raw = base_eff - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - base_eff
    direction = np.clip(direction_raw / 300.0, -1.0, 1.0)
    direction_ch = np.full(WINDOW, direction, dtype=np.float32)
    return np.stack([norm, d1, d2, direction_ch]).astype(np.float32)


def label_scenario_window(sig_w, chg_w, diff_w, grad_w, floor_type):
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    abs_grad = np.abs(grad_w)
    max_change = float(np.max(chg_w))
    avg_change = float(np.mean(chg_w))
    max_gradient = float(np.max(abs_grad))
    edge_ratio = float(np.mean(abs_grad >= params['gradient']))
    mean_floor_diff = float(np.mean(np.abs(diff_w)))
    median_floor_diff = float(np.abs(np.median(diff_w)))
    sign_consistency = float(max(np.mean(diff_w >= 0), np.mean(diff_w <= 0)))
    flat_std = float(np.std(sig_w))
    noise_ratio, spike_ratio = compute_noise_metrics(diff_w, sig_w)

    hard_floor = (
        max_change < params['change_low'] * 0.90 and
        mean_floor_diff < params['clear_mean'] * 1.05 and
        max_gradient < params['gradient'] * 0.95 and
        flat_std < max(2.2, params['clear_mean'] * 1.2)
    )

    event_liquid = (
        max_change >= params['change_low'] and
        max_gradient >= params['gradient'] * 1.05 and
        edge_ratio >= 0.18
    )

    attach_plateau = (
        max_change >= params['change_low'] and
        mean_floor_diff >= params['plateau_mean'] and
        median_floor_diff >= params['plateau_med'] and
        sign_consistency >= params['sign_keep'] and
        noise_ratio <= params.get('noise_ratio', 0.8) * 1.05 and
        spike_ratio <= params.get('spike_ratio', 2.0) * 1.05
    )

    if hard_floor:
        return '바닥'
    if event_liquid:
        return '액체'
    if attach_plateau and avg_change >= params['change_low'] * 1.05:
        return '액체'
    return None


def add_manual_sample(signal, base, floor_type, label, X_all, y_all, repeat=3):
    sig = resize_signal(signal, WINDOW)
    full_signal = sig.copy()
    win = build_window(sig, base, floor_type, full_signal, 0)
    for _ in range(repeat):
        aug = win + np.random.normal(0, random.uniform(0.008, 0.03), win.shape)
        aug[:3] *= random.uniform(0.97, 1.03)
        X_all.append(aug.astype(np.float32))
        y_all.append(label)


def build_model():
    inp = Input(shape=(WINDOW, 4))
    x = Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.20)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.30)(x)
    out = Dense(len(CLASSES), activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def write_log(log_path, lines):
    with open(log_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(lines))


def collect_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, '*.csv')))


def train_models(data_dir=DATA_DIR, manual_dir=MANUAL_DIR, save_dir=SAVE_DIR):
    encoder = LabelEncoder().fit(CLASSES)
    with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_lines = [
        '2-4 binary event-gate train log',
        '=' * 90,
        f'[DATA_DIR] {data_dir}',
        f'[MANUAL_DIR] {manual_dir}',
        f'[SAVE_DIR] {os.path.abspath(save_dir)}',
        f'[encoder] {os.path.abspath(os.path.join(save_dir, "encoder.pkl"))}',
        ''
    ]

    floor_files = {f: {'scenario': [], 'manual': []} for f in FLOOR_KEYWORDS}

    for fp in collect_files(data_dir):
        floor = detect_floor_type(os.path.basename(fp))
        if floor:
            floor_files[floor]['scenario'].append(fp)

    if os.path.isdir(manual_dir):
        for fp in collect_files(manual_dir):
            floor = detect_floor_type(os.path.basename(fp))
            if floor:
                floor_files[floor]['manual'].append(fp)

    for floor, groups in floor_files.items():
        X_all, y_all = [], []
        log_lines.append(f'\n[바닥] {floor}')
        log_lines.append(f'  scenario={len(groups["scenario"])} | manual={len(groups["manual"])}')

        # 1) scenario full files
        for fp in groups['scenario']:
            base, signal, changes = read_csv_recalc_change(fp)
            if base is None:
                log_lines.append(f'  [SKIP scenario read fail] {os.path.basename(fp)}')
                continue

            smooth_sig, grad_sig, floor_ref = build_recent_floor_reference(signal, base, floor)
            max_start = len(signal) - WINDOW
            if max_start < 0:
                log_lines.append(f'  [SKIP short] {os.path.basename(fp)}')
                continue

            added = 0
            for start in range(0, max_start + 1, STEP_SCENARIO):
                sig_w = smooth_sig[start:start + WINDOW]
                chg_w = np.abs(sig_w - base).astype(np.float32)
                diff_w = sig_w - floor_ref[start:start + WINDOW]
                grad_w = grad_sig[start:start + WINDOW]
                label = label_scenario_window(sig_w, chg_w, diff_w, grad_w, floor)
                if label is None:
                    continue
                win = build_window(sig_w, base, floor, smooth_sig, start)
                repeat = 3 if label == '액체' else 2
                for _ in range(repeat):
                    aug = win + np.random.normal(0, random.uniform(0.01, 0.04), win.shape)
                    aug[:3] *= random.uniform(0.95, 1.05)
                    X_all.append(aug.astype(np.float32))
                    y_all.append(label)
                added += repeat
            log_lines.append(f'  [SCENARIO] {os.path.basename(fp)} | len={len(signal)} | base={base:.2f} | +{added}')

        # 2) manual cropped files
        for fp in groups['manual']:
            mode = detect_manual_mode(os.path.basename(fp))
            base, signal, changes = read_csv_recalc_change(fp)
            if base is None:
                log_lines.append(f'  [SKIP manual read fail] {os.path.basename(fp)}')
                continue
            if mode == 'liquid':
                add_manual_sample(signal, base, floor, '액체', X_all, y_all, repeat=5)
                # 앞뒤 일부를 바닥 negative로 추가 시도
                if len(signal) >= 8:
                    edge_len = max(4, len(signal) // 4)
                    add_manual_sample(signal[:edge_len], base, floor, '바닥', X_all, y_all, repeat=2)
                    add_manual_sample(signal[-edge_len:], base, floor, '바닥', X_all, y_all, repeat=2)
                log_lines.append(f'  [MANUAL liquid] {os.path.basename(fp)} | len={len(signal)} | base={base:.2f} | +9')
            elif mode == 'floor':
                add_manual_sample(signal, base, floor, '바닥', X_all, y_all, repeat=5)
                log_lines.append(f'  [MANUAL floor] {os.path.basename(fp)} | len={len(signal)} | base={base:.2f} | +5')
            else:
                # scenario-like file in manual dir: conservative label extraction
                smooth_sig, grad_sig, floor_ref = build_recent_floor_reference(signal, base, floor)
                max_start = len(signal) - WINDOW
                added = 0
                for start in range(0, max_start + 1, max(1, STEP_SCENARIO)):
                    sig_w = smooth_sig[start:start + WINDOW]
                    chg_w = np.abs(sig_w - base).astype(np.float32)
                    diff_w = sig_w - floor_ref[start:start + WINDOW]
                    grad_w = grad_sig[start:start + WINDOW]
                    label = label_scenario_window(sig_w, chg_w, diff_w, grad_w, floor)
                    if label is None:
                        continue
                    win = build_window(sig_w, base, floor, smooth_sig, start)
                    repeat = 3 if label == '액체' else 2
                    for _ in range(repeat):
                        aug = win + np.random.normal(0, random.uniform(0.01, 0.04), win.shape)
                        aug[:3] *= random.uniform(0.95, 1.05)
                        X_all.append(aug.astype(np.float32))
                        y_all.append(label)
                    added += repeat
                log_lines.append(f'  [MANUAL scenario] {os.path.basename(fp)} | len={len(signal)} | base={base:.2f} | +{added}')

        if len(X_all) == 0:
            log_lines.append('  [SKIP] 학습 데이터 없음')
            continue

        X = np.array(X_all, dtype=np.float32).transpose(0, 2, 1)
        y = encoder.transform(y_all)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))
        classes_present = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, class_weights)}

        uniq, cnt = np.unique(y_all, return_counts=True)
        label_dist = {u: int(c) for u, c in zip(uniq, cnt)}
        log_lines.append(f'  [LABEL_DIST] {label_dist}')

        model = build_model()
        save_path = os.path.join(save_dir, f'model_{floor}.keras')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ]

        print(f'[TRAIN] 바닥={floor} | 샘플수={len(X)} | 라벨={label_dist}')
        hist = model.fit(
            X, y_cat,
            validation_split=0.2,
            epochs=60,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )
        best_val = float(np.min(hist.history.get('val_loss', [0.0])))
        log_lines.append(f'  [MODEL_SAVE] {os.path.abspath(save_path)}')
        log_lines.append(f'  [BEST_VAL_LOSS] {best_val:.6f}')

    log_path = os.path.join(save_dir, f'train_log_2-4_binary_eventgate_v1_{ts}.txt')
    write_log(log_path, log_lines)
    print(f'\n[학습 로그 저장] {os.path.abspath(log_path)}')
    print(f'[encoder 저장] {os.path.abspath(os.path.join(save_dir, "encoder.pkl"))}')


if __name__ == '__main__':
    train_models()
