# -*- coding: utf-8 -*-
"""
2-4 구조 유지 + binary(바닥/액체) + event gate 중심 학습 코드

핵심
- 4채널(norm, d1, d2, direction) 유지
- 출력 라벨은 ['바닥', '액체']
- 액체 plateau를 길게 유지하도록 학습하지 않음
- rise/fall event 근처만 액체로 사용
- 수동 _rise/_fall: 액체 / _flat/_floor: 바닥
- 결과 로그를 SAVE_DIR에 저장
"""

import os, glob, csv, pickle, random, re, datetime
from collections import Counter
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR = r"C:\Users\MASL\Desktop\코드개선\3차 수동\3rd_수동"
SAVE_DIR = "./kumoh_lstm_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 15
SMOOTH_K = 5

BINARY_CLASSES = ['바닥', '액체']

FLOOR_KEYWORDS = {
    '검대': ['검대'], '회대': ['회대'], '황대': ['황대'],
    '207회바': ['greyfloor', '회색바닥'], '흰책상': ['white', '하양', '흰', 'whitedesk'],
    '나타': ['나타'], '회타': ['회타'], '나무': ['나무', 'wood']
}
REVERSE_DIRECTION_FLOORS = {'검대'}
RAW_RE = re.compile(r"Time=(?P<time>[-+]?\d+(?:\.\d+)?)s\s*\|\s*BaseRaw=(?P<base>[-+]?\d+(?:\.\d+)?)\s*\|\s*CurrentRaw=(?P<current>[-+]?\d+(?:\.\d+)?)\s*\|\s*Change=(?P<change>[-+]?\d+(?:\.\d+)?)")

# per-floor conservative thresholds based on 2-4
FLOOR_THRESHOLDS = {
    '나무': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.030, 'ref_grad': 0.80, 'ref_band': 6.5},
    '황대': {'gradient': 0.6, 'change_low': 5.0, 'change_strong': 15.0, 'plateau_mean': 7.0, 'plateau_med': 6.0, 'clear_mean': 2.3, 'alpha': 0.028, 'ref_grad': 0.75, 'ref_band': 6.0},
    '회대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0},
    '검대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0},
    '207회바': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0},
    '흰책상': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0},
    '나타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'plateau_mean': 10.0, 'plateau_med': 8.0, 'clear_mean': 3.0, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0},
    '회타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'plateau_mean': 10.0, 'plateau_med': 8.0, 'clear_mean': 3.0, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0},
}


def to_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def smooth_signal(x, k=SMOOTH_K):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, ker, mode='same')


def remove_spikes(signal, k=3.0, local=5):
    sig = np.array(signal, dtype=np.float32).copy()
    if len(sig) < 3:
        return sig
    for i in range(1, len(sig)-1):
        l = max(0, i-local); r = min(len(sig), i+local+1)
        med = float(np.median(sig[l:r]))
        std = float(np.std(sig[l:r])) + 1e-6
        if abs(float(sig[i])-med) > k*std and abs(float(sig[i])-float(sig[i-1])) > 1.2*std and abs(float(sig[i])-float(sig[i+1])) > 1.2*std:
            sig[i] = np.float32(0.5*(sig[i-1]+sig[i+1]))
    return sig


def detect_floor_type(filename):
    lower = filename.lower()
    for floor, kws in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return floor
    return None


def is_manual_event(filename):
    lower = filename.lower()
    return ('_rise' in lower) or ('_fall' in lower)


def is_manual_floor(filename):
    lower = filename.lower()
    return ('_flat' in lower) or ('_floor' in lower)


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
        if len(currents) == 0:
            return None, None, None
        base = float(np.mean(baseline_values)) if baseline_values else (float(np.mean(row_bases)) if row_bases else None)
        if base is None:
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
                base_raw = None; current_raw = None
                for i in range(len(parts)-1):
                    if parts[i] == 'BaseRaw':
                        base_raw = to_float(parts[i+1])
                    elif parts[i] == 'CurrentRaw':
                        current_raw = to_float(parts[i+1])
                if current_raw is not None:
                    currents.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)
        if len(currents) == 0:
            return None, None, None
        base = final_base if final_base is not None else (float(np.mean(baseline_values)) if baseline_values else (float(np.mean(row_bases)) if row_bases else None))
        if base is None:
            return None, None, None
        currents = np.array(currents, dtype=np.float32)
        changes = np.abs(currents - base).astype(np.float32)
        return base, currents, changes

    for enc in encodings:
        try:
            with open(fp, 'r', encoding=enc, errors='ignore', newline='') as f:
                lines = f.readlines()
            out = parse_new(lines)
            if out[0] is not None:
                return out
            out = parse_old(lines)
            if out[0] is not None:
                return out
        except Exception:
            continue
    return None, None, None


def resample_to_window(signal, target_len=WINDOW_SIZE):
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) == target_len:
        return signal
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, signal).astype(np.float32)


def build_recent_floor_reference(signal, baseline, floor_type):
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    sig = smooth_signal(remove_spikes(signal))
    grad = np.gradient(sig)
    ref = np.empty_like(sig)
    ref[0] = baseline
    alpha, ref_grad, ref_band = params['alpha'], params['ref_grad'], params['ref_band']
    for i in range(1, len(sig)):
        stable_grad = abs(grad[i]) < ref_grad
        close_to_ref = abs(sig[i] - ref[i-1]) < ref_band
        if stable_grad and close_to_ref:
            ref[i] = (1-alpha)*ref[i-1] + alpha*sig[i]
        else:
            ref[i] = ref[i-1]
    return sig, grad, ref


def create_window(signal, start, base, floor_type):
    sig = signal[start:start+WINDOW_SIZE]
    if len(sig) != WINDOW_SIZE:
        return None
    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3)/3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])
    base_eff = base if len(signal) < 20 else 0.7*base + 0.3*np.mean(signal[:20])
    ctx_start = max(0, start-5); ctx_end = min(len(signal), start+WINDOW_SIZE+5)
    ctx_mean = np.mean(signal[ctx_start:ctx_end])
    direction_raw = base_eff - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - base_eff
    direction = np.clip(direction_raw/300.0, -1.0, 1.0)
    direction_ch = np.full(WINDOW_SIZE, direction, dtype=np.float32)
    return np.stack([norm, d1, d2, direction_ch]).astype(np.float32)


def window_features(sig_w, chg_w, diff_w, grad_w, params):
    max_change = float(np.max(chg_w))
    mean_floor_diff = float(np.mean(np.abs(diff_w)))
    median_floor_diff = float(np.abs(np.median(diff_w)))
    max_grad = float(np.max(np.abs(grad_w)))
    edge_ratio = float(np.mean(np.abs(grad_w) >= params['gradient']))
    flat_std = float(np.std(sig_w))
    event_seed = (max_grad >= params['gradient'] * 1.8 and max_change >= params['change_low']) or (edge_ratio >= 0.40 and max_change >= params['change_low'])
    attach_plateau = (mean_floor_diff >= params['plateau_mean'] and median_floor_diff >= params['plateau_med'] and max_change >= params['change_low'] and flat_std < 20.0)
    hard_floor = (max_change < params['change_low'] * 0.90 and mean_floor_diff < params['clear_mean'] and max_grad < params['gradient'] * 0.90)
    return dict(max_change=max_change, mean_floor_diff=mean_floor_diff, median_floor_diff=median_floor_diff,
                max_grad=max_grad, edge_ratio=edge_ratio, flat_std=flat_std,
                event_seed=event_seed, attach_plateau=attach_plateau, hard_floor=hard_floor)


def build_model():
    inp = Input(shape=(WINDOW_SIZE, 4))
    x = Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    out = Dense(len(BINARY_CLASSES), activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_models(data_dir=DATA_DIR, save_dir=SAVE_DIR):
    log_lines = []
    encoder = LabelEncoder().fit(BINARY_CLASSES)
    with open(os.path.join(save_dir, 'encoder_binary.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

    floor_files = {f: [] for f in FLOOR_KEYWORDS}
    for fp in glob.glob(os.path.join(data_dir, '*.csv')):
        floor = detect_floor_type(os.path.basename(fp))
        if floor:
            floor_files[floor].append(fp)

    for floor, file_list in floor_files.items():
        if not file_list:
            msg = f'[SKIP] {floor}: 파일 없음'
            print(msg); log_lines.append(msg)
            continue
        params = FLOOR_THRESHOLDS.get(floor, FLOOR_THRESHOLDS['회대'])
        X_all, y_all = [], []
        counts = Counter()
        for fp in file_list:
            name = os.path.basename(fp)
            manual_event = is_manual_event(name)
            manual_floor = is_manual_floor(name)
            base, signal, changes = read_csv_recalc_change(fp)
            if base is None:
                continue
            signal = np.asarray(signal, dtype=np.float32)
            if len(signal) < WINDOW_SIZE:
                # short manual file support
                signal = resample_to_window(signal, WINDOW_SIZE)
                changes = np.abs(signal - float(base)).astype(np.float32)
            smooth_sig, grad_sig, floor_ref = build_recent_floor_reference(signal, base, floor)
            max_start = len(signal) - WINDOW_SIZE
            starts = [0] if max_start <= 0 else list(range(0, max_start + 1, 1 if manual_event or manual_floor else max(1, max_start // 25)))
            feats = []
            wins = []
            for st in starts:
                win = create_window(smooth_sig, st, base, floor)
                if win is None:
                    continue
                sig_w = smooth_sig[st:st+WINDOW_SIZE]
                chg_w = changes[st:st+WINDOW_SIZE]
                ref_w = floor_ref[st:st+WINDOW_SIZE]
                diff_w = sig_w - ref_w
                grad_w = grad_sig[st:st+WINDOW_SIZE]
                f = window_features(sig_w, chg_w, diff_w, grad_w, params)
                feats.append(f); wins.append(win)
            if not wins:
                continue
            # label windows conservatively
            labels = [None] * len(wins)
            if manual_event:
                for i, f in enumerate(feats):
                    labels[i] = '액체' if (f['event_seed'] or f['attach_plateau']) else None
            elif manual_floor:
                for i in range(len(wins)):
                    labels[i] = '바닥'
            else:
                seeds = [i for i, f in enumerate(feats) if f['event_seed']]
                seed_mask = np.zeros(len(wins), dtype=bool)
                for i in seeds:
                    seed_mask[max(0, i-1):min(len(wins), i+2)] = True  # short bridge only
                for i, f in enumerate(feats):
                    if f['hard_floor']:
                        labels[i] = '바닥'
                    elif seed_mask[i] and (f['event_seed'] or f['attach_plateau']):
                        labels[i] = '액체'
                    elif not seed_mask[i] and f['hard_floor']:
                        labels[i] = '바닥'
                    else:
                        labels[i] = None
            for win, lab in zip(wins, labels):
                if lab is None:
                    continue
                aug_count = 3 if lab == '바닥' else 2
                counts[lab] += aug_count
                for _ in range(aug_count):
                    aug = win + np.random.normal(0, random.uniform(0.01, 0.03), win.shape)
                    aug[:3] *= random.uniform(0.97, 1.03)
                    X_all.append(aug.astype(np.float32))
                    y_all.append(lab)
        if len(X_all) == 0:
            msg = f'[SKIP] {floor}: 학습 데이터가 없습니다.'
            print(msg); log_lines.append(msg)
            continue
        X = np.array(X_all, dtype=np.float32).transpose(0, 2, 1)
        y = encoder.transform(y_all)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(BINARY_CLASSES))
        classes_present = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, class_weights)}
        model = build_model()
        save_path = os.path.join(save_dir, f'model_{floor}_binary.keras')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ]
        msg = f'[TRAIN] 바닥={floor} | 파일수={len(file_list)} | 샘플수={len(X)} | 분포={dict(counts)}'
        print(msg); log_lines.append(msg)
        model.fit(X, y_cat, validation_split=0.2, epochs=60, batch_size=32, callbacks=callbacks, class_weight=class_weight_dict, verbose=1)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(save_dir, f'train_2-4_binary_eventgate_v1_{ts}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f'학습 완료 / 로그 저장: {log_path}')


if __name__ == '__main__':
    train_models()
