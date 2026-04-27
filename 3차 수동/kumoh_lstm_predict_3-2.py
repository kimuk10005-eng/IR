# -*- coding: utf-8 -*-
"""
기존 2-4 구조 유지 + 로컬 베이스라인 + Moving Average 반영 예측 코드
"""

import os
import csv
import re
import pickle
from collections import Counter

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = './kumoh_lstm_model_save'
ENCODER_PATH = os.path.join(SAVE_DIR, 'encoder.pkl')

WINDOW = 15
STEP = 5
SMOOTH_K = 5
MOVING_AVG_K = 7
LOCAL_BASELINE_MARGIN = 12
LOCAL_BASELINE_MIN_POINTS = 5

VOTE_RADIUS = 1
MIN_LIQUID_RUN = 2
EVENT_SCORE_THRESHOLD = 0.52

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

FLOOR_MODELS = {
    '나무': os.path.join(SAVE_DIR, 'model_나무.keras'),
    '황대': os.path.join(SAVE_DIR, 'model_황대.keras'),
    '회대': os.path.join(SAVE_DIR, 'model_회대.keras'),
    '검대': os.path.join(SAVE_DIR, 'model_검대.keras'),
    '그마': os.path.join(SAVE_DIR, 'model_그마.keras'),
    '207회바': os.path.join(SAVE_DIR, 'model_207회바.keras'),
    '흰책상': os.path.join(SAVE_DIR, 'model_흰책상.keras'),
    '나타': os.path.join(SAVE_DIR, 'model_나타.keras'),
    '회타': os.path.join(SAVE_DIR, 'model_회타.keras')
}

FLOOR_ALIASES = {
    '나무': ['나무', 'wood', '나'],
    '황대': ['황대', '황색대리석', '황색', 'yellow', '황'],
    '회대': ['회대', '회색대리석', '회색', 'gray', 'grey', '회'],
    '검대': ['검대', '검정색대리석', '검정', 'black', '검'],
    '그마': ['그마', 'greymarble'],
    '207회바': ['회바', '207', '207greyfloor', 'greyfloor'],
    '흰책상': ['흰책상', 'white', 'whitedesk'],
    '나타': ['나타'],
    '회타': ['회타'],
}

REVERSE_DIRECTION_FLOORS = {'검대'}
FLOOR_COLOR = '#e0e0e0'
LIQUID_COLOR = '#7b4a1e'

RAW_RE = re.compile(
    r"Time=(?P<time>[-+]?\d+(?:\.\d+)?)s\s*\|\s*"
    r"BaseRaw=(?P<base>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"CurrentRaw=(?P<current>[-+]?\d+(?:\.\d+)?)\s*\|\s*"
    r"Change=(?P<change>[-+]?\d+(?:\.\d+)?)"
)

try:
    encoder = pickle.load(open(ENCODER_PATH, 'rb'))
    classes = encoder.classes_
except Exception as e:
    print(f"인코더 로드 실패: {e}")
    raise SystemExit

models = {}
for floor, path in FLOOR_MODELS.items():
    if os.path.exists(path):
        try:
            models[floor] = tf.keras.models.load_model(path)
            print(f"모델 로드 완료: {floor}")
        except Exception as e:
            print(f"{floor} 모델 로드 실패: {e}")
    else:
        print(f"모델 파일 없음: {path}")

def to_float(x):
    try:
        x = str(x).strip()
        return float(x) if x != '' else None
    except Exception:
        return None

def find_floor_type(text):
    text = text.lower().strip()
    for floor, aliases in FLOOR_ALIASES.items():
        if text in [a.lower() for a in aliases]:
            return floor
    return None

def smooth_signal(x, k=SMOOTH_K):
    if len(x) == 0:
        return x
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode='same')

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
        if abs(float(sig[i]) - local_med) > k * local_std and abs(float(sig[i]) - float(sig[i-1])) > 1.2 * local_std and abs(float(sig[i]) - float(sig[i+1])) > 1.2 * local_std:
            sig[i] = np.float32(0.5 * (sig[i - 1] + sig[i + 1]))
    return sig

def compute_noise_metrics(diff_w, smooth_w):
    mean_abs_diff = float(np.mean(np.abs(diff_w))) + 1e-6
    noise_ratio = float(np.std(diff_w) / mean_abs_diff)
    spike_ratio = float((np.max(smooth_w) - np.min(smooth_w)) / mean_abs_diff)
    return noise_ratio, spike_ratio

def read_csv_with_fixed_baseline(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    def parse_new_format(lines):
        baseline_values = []
        row_bases = []
        data_points = []
        timestamps = []

        reader = csv.DictReader(lines)
        fields = [str(name).strip() for name in (reader.fieldnames or [])]
        if 'record_type' not in fields:
            return None, None, None, None

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
                    data_points.append(current_raw)
                    m = re.search(r'Time\s*=\s*([0-9.]+)s', raw_data)
                    timestamps.append(m.group(1) + 's' if m else str(len(data_points) - 1))
                if base_raw is not None:
                    row_bases.append(base_raw)

        if len(data_points) < WINDOW:
            return None, None, None, None

        if baseline_values:
            base = float(np.mean(baseline_values))
        elif row_bases:
            base = float(np.mean(row_bases))
        else:
            return None, None, None, None

        buffer = np.array(data_points, dtype=np.float32)
        changes = np.abs(buffer - base).astype(np.float32)
        return base, buffer, changes, timestamps

    def parse_old_format(lines):
        baseline_values = []
        row_bases = []
        data_points = []
        timestamps = []
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
                time_val = None
                base_raw = None
                current_raw = None
                if len(parts) >= 2:
                    time_val = parts[1]

                for i in range(len(parts) - 1):
                    key = parts[i]
                    val = parts[i + 1]
                    if key == 'BaseRaw':
                        base_raw = to_float(val)
                    elif key == 'CurrentRaw':
                        current_raw = to_float(val)

                if current_raw is not None:
                    data_points.append(current_raw)
                    timestamps.append(time_val if time_val else str(len(data_points) - 1))
                if base_raw is not None:
                    row_bases.append(base_raw)

        if len(data_points) < WINDOW:
            return None, None, None, None

        if final_base is not None:
            base = float(final_base)
        elif baseline_values:
            base = float(np.mean(baseline_values))
        elif row_bases:
            base = float(np.mean(row_bases))
        else:
            return None, None, None, None

        buffer = np.array(data_points, dtype=np.float32)
        changes = np.abs(buffer - base).astype(np.float32)
        return base, buffer, changes, timestamps

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

    return None, None, None, None

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

def compute_local_baseline(signal, start, window_size=WINDOW):
    left = max(0, start - LOCAL_BASELINE_MARGIN)
    right = max(0, start - 1)
    if right - left + 1 >= LOCAL_BASELINE_MIN_POINTS:
        return float(np.mean(signal[left:right + 1]))
    left2 = start + window_size
    right2 = min(len(signal) - 1, start + window_size + LOCAL_BASELINE_MARGIN - 1)
    if right2 - left2 + 1 >= LOCAL_BASELINE_MIN_POINTS:
        return float(np.mean(signal[left2:right2 + 1]))
    return None

def create_window(full_signal, start_idx, baseline, floor_type, local_base=None):
    if start_idx + WINDOW > len(full_signal):
        return None
    sig = full_signal[start_idx:start_idx + WINDOW]
    norm_sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    deriv1 = np.diff(norm_sig)
    deriv1 = np.convolve(deriv1, np.ones(3) / 3, 'same')
    deriv1 = np.pad(deriv1, (1, 0))
    deriv2 = np.diff(deriv1)
    deriv2 = np.pad(deriv2, (1, 0))

    if local_base is None:
        local_base = baseline if len(full_signal) < 20 else 0.7 * baseline + 0.3 * np.mean(full_signal[:20])

    context_start = max(0, start_idx - 5)
    context_end = min(len(full_signal), start_idx + WINDOW + 5)
    recent_mean = np.mean(full_signal[context_start:context_end])
    change_raw = local_base - recent_mean if floor_type in REVERSE_DIRECTION_FLOORS else recent_mean - local_base
    direction = np.clip(change_raw / 300.0, -1.0, 1.0)
    direction_channel = np.full((WINDOW,), direction)
    window = np.stack([norm_sig, deriv1, deriv2, direction_channel])
    return window[np.newaxis, :, :]

def get_best_non_floor_label(pred):
    floor_idx = np.where(classes == '바닥')[0][0] if '바닥' in classes else -1
    temp = pred.copy()
    if floor_idx >= 0:
        temp[floor_idx] = -1.0
    idx = int(np.argmax(temp))
    return classes[idx], float(max(temp[idx] * 100.0, 0.0))

def sigmoid01(x, center, width):
    width = max(width, 1e-6)
    return 1.0 / (1.0 + np.exp(-(x - center) / width))

def compute_event_score(max_change, ma_change, max_gradient, ma_gradient, mean_floor_diff, noise_ratio, spike_ratio, params):
    s_change = sigmoid01(max_change, params['change_low'] * 0.95, 1.6)
    s_ma_change = sigmoid01(ma_change, params['change_low'] * 0.75, 1.2)
    s_grad = sigmoid01(max_gradient, params['gradient'] * 1.05, 0.18)
    s_ma_grad = sigmoid01(ma_gradient, params['gradient'] * 0.90, 0.15)
    s_floor_diff = sigmoid01(mean_floor_diff, params['clear_mean'] + 1.0, 1.0)

    penalty_noise = 1.0 if noise_ratio <= params.get('noise_ratio', 0.8) * 1.1 else 0.65
    penalty_spike = 1.0 if spike_ratio <= params.get('spike_ratio', 2.0) * 1.1 else 0.7

    score = (
        0.28 * s_change +
        0.22 * s_ma_change +
        0.22 * s_grad +
        0.18 * s_ma_grad +
        0.10 * s_floor_diff
    ) * penalty_noise * penalty_spike

    return float(np.clip(score, 0.0, 1.0))

def apply_vote_smoothing(predictions):
    if not predictions:
        return predictions
    labels = [p['label'] for p in predictions]
    new_labels = labels[:]

    for i in range(len(predictions)):
        left = max(0, i - VOTE_RADIUS)
        right = min(len(predictions), i + VOTE_RADIUS + 1)
        neigh = labels[left:right]
        floor_count = sum(1 for x in neigh if x == '바닥')
        liquid_count = len(neigh) - floor_count
        if floor_count > liquid_count:
            new_labels[i] = '바닥'
        elif liquid_count > floor_count and labels[i] != '바닥':
            new_labels[i] = labels[i]

    for i, p in enumerate(predictions):
        p['label'] = new_labels[i]
    return predictions

def apply_conservative_postprocess(predictions, floor_type):
    if not predictions:
        return predictions

    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])

    for p in predictions:
        flat_cond = (
            p['ma_change'] < params['change_low'] * 0.85
            and p['ma_gradient'] < params['gradient'] * 0.85
            and p['mean_floor_diff'] < params['clear_mean'] * 1.10
        )
        weak_event_cond = (
            p['event_score'] < EVENT_SCORE_THRESHOLD
            and p['max_gradient'] < params['gradient'] * 1.10
        )
        plateau_only_cond = (
            p['mean_floor_diff'] >= params['plateau_mean']
            and p['ma_gradient'] < params['gradient'] * 0.95
            and p['max_gradient'] < params['gradient'] * 1.20
        )

        if flat_cond or weak_event_cond or plateau_only_cond:
            p['label'] = '바닥'
            p['prob'] = max(float(p['prob']), 82.0)

    predictions = apply_vote_smoothing(predictions)

    labels = [p['label'] for p in predictions]
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] == '바닥':
            i += 1
            continue

        j = i
        while j + 1 < n and labels[j + 1] != '바닥':
            j += 1

        run_len = j - i + 1
        run_score = max(predictions[k]['event_score'] for k in range(i, j + 1))
        run_ma_grad = max(predictions[k]['ma_gradient'] for k in range(i, j + 1))
        run_ma_change = max(predictions[k]['ma_change'] for k in range(i, j + 1))

        if (
            run_len < MIN_LIQUID_RUN
            or run_score < EVENT_SCORE_THRESHOLD + 0.06
            or run_ma_grad < params['gradient']
            or run_ma_change < params['change_low']
        ):
            for k in range(i, j + 1):
                predictions[k]['label'] = '바닥'
                predictions[k]['prob'] = max(float(predictions[k]['prob']), 80.0)
        i = j + 1

    return predictions

def predict(buffer, changes, baseline, floor_type, model):
    predictions = []
    max_start = len(buffer) - WINDOW
    if max_start < 0:
        return []

    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    clean_buffer = remove_spikes(buffer)
    sig_smooth, change_gradient, floor_ref = build_recent_floor_reference(clean_buffer, baseline, floor_type)

    ma_grad_global = moving_average(np.abs(change_gradient), MOVING_AVG_K)
    floor_idx = np.where(classes == '바닥')[0][0] if '바닥' in classes else -1

    for start_idx in range(0, max_start + 1, STEP):
        local_base = compute_local_baseline(sig_smooth, start_idx, WINDOW)
        if local_base is None:
            local_base = baseline

        local_changes_full = np.abs(sig_smooth - local_base).astype(np.float32)
        ma_change_global = moving_average(local_changes_full, MOVING_AVG_K)

        window = create_window(sig_smooth, start_idx, baseline, floor_type, local_base=local_base)
        if window is None:
            continue

        smooth_w = sig_smooth[start_idx:start_idx + WINDOW]
        ref_w = floor_ref[start_idx:start_idx + WINDOW]
        diff_w = smooth_w - ref_w
        window_changes = local_changes_full[start_idx:start_idx + WINDOW]
        window_gradient = change_gradient[start_idx:start_idx + WINDOW]
        ma_change_w = ma_change_global[start_idx:start_idx + WINDOW]
        ma_grad_w = ma_grad_global[start_idx:start_idx + WINDOW]

        avg_change = float(np.mean(window_changes))
        max_change = float(np.max(window_changes))
        max_gradient = float(np.max(np.abs(window_gradient)))
        ma_change = float(np.mean(ma_change_w))
        ma_gradient = float(np.mean(ma_grad_w))
        mean_floor_diff = float(np.mean(np.abs(diff_w)))
        median_floor_diff = float(np.abs(np.median(diff_w)))
        sign_consistency = float(max(np.mean(diff_w >= 0), np.mean(diff_w <= 0)))
        noise_ratio, spike_ratio = compute_noise_metrics(diff_w, smooth_w)

        if model:
            pred = model.predict(np.transpose(window, (0, 2, 1)), verbose=0)[0]
        else:
            pred = np.zeros(len(classes), dtype=np.float32)
            pred[floor_idx] = 1.0

        orig_idx = int(np.argmax(pred))
        orig_label = classes[orig_idx]
        orig_prob = float(pred[orig_idx] * 100.0)
        best_non_floor_label, best_non_floor_prob = get_best_non_floor_label(pred)

        simple_liquid = max_change >= params['change_low']
        strong_liquid = max_change >= params['change_strong']
        hard_floor = (
            max_change < params['change_low'] * 0.90
            and mean_floor_diff < params['clear_mean']
            and max_gradient < params['gradient'] * 0.90
        )
        plateau_flag = (
            mean_floor_diff >= params['plateau_mean']
            and median_floor_diff >= params['plateau_med']
            and sign_consistency >= params['sign_keep']
            and noise_ratio <= params.get('noise_ratio', 0.8)
            and spike_ratio <= params.get('spike_ratio', 2.0)
        )
        very_clear_plateau = (
            mean_floor_diff >= params['plateau_mean'] + 1.8
            and median_floor_diff >= params['plateau_med'] + 1.2
            and sign_consistency >= 0.92
            and noise_ratio <= params.get('noise_ratio', 0.8) * 0.9
            and spike_ratio <= params.get('spike_ratio', 2.0) * 0.9
        )

        event_score = compute_event_score(
            max_change=max_change,
            ma_change=ma_change,
            max_gradient=max_gradient,
            ma_gradient=ma_gradient,
            mean_floor_diff=mean_floor_diff,
            noise_ratio=noise_ratio,
            spike_ratio=spike_ratio,
            params=params,
        )

        if hard_floor:
            final_label = '바닥'
            final_prob = 95.0
        elif strong_liquid and event_score >= 0.68 and orig_label != '바닥' and orig_prob >= 68.0:
            final_label = orig_label
            final_prob = max(orig_prob, 82.0)
        elif strong_liquid and event_score >= 0.68:
            final_label = best_non_floor_label
            final_prob = max(best_non_floor_prob, 76.0)
        elif simple_liquid and very_clear_plateau and event_score >= 0.60 and best_non_floor_prob >= 40.0:
            final_label = best_non_floor_label
            final_prob = max(best_non_floor_prob, 70.0)
        elif simple_liquid and plateau_flag and event_score >= 0.58 and orig_label != '바닥' and orig_prob >= 72.0:
            final_label = orig_label
            final_prob = max(orig_prob, 76.0)
        else:
            final_label = '바닥'
            final_prob = max(62.0, 100.0 - avg_change)

        predictions.append({
            'start': start_idx,
            'end': start_idx + WINDOW,
            'label': final_label,
            'prob': final_prob,
            'orig_label': orig_label,
            'orig_prob': orig_prob,
            'best_non_floor_label': best_non_floor_label,
            'best_non_floor_prob': best_non_floor_prob,
            'avg_change': avg_change,
            'max_change': max_change,
            'max_gradient': max_gradient,
            'ma_change': ma_change,
            'ma_gradient': ma_gradient,
            'event_score': event_score,
            'mean_floor_diff': mean_floor_diff,
            'median_floor_diff': median_floor_diff,
            'sign_consistency': sign_consistency,
            'noise_ratio': noise_ratio,
            'spike_ratio': spike_ratio,
            'local_baseline': float(local_base),
        })

    return apply_conservative_postprocess(predictions, floor_type)

def visualize(predictions, buffer, changes, baseline, floor_type):
    if not predictions:
        print('시각화할 예측 결과가 없습니다.')
        return

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.0, 2.2, 1.4, 1.8], hspace=0.30)
    fig.suptitle(f'{floor_type} 바닥 - LSTM 실시간 액체 감지 결과', fontsize=20, fontweight='bold', y=0.97)

    ax1 = fig.add_subplot(gs[0])
    for r in predictions:
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        ax1.axvspan(r['start'], r['end'], color=color, alpha=0.22)

    ax1.plot(buffer, color='blue', lw=2, label='CurrentRaw')
    ax1.axhline(baseline, color='green', lw=2, ls='--', label=f'Base: {baseline:.1f}')
    ax1.grid(alpha=0.25)
    ax1.legend(loc='upper left', fontsize=11)

    ax1r = ax1.twinx()
    ax1r.plot(changes, color='red', lw=2, ls=':', alpha=0.75, label='Global Change')
    ax1r.legend(loc='upper right', fontsize=11)

    ax2 = fig.add_subplot(gs[1])
    probs = [r['prob'] for r in predictions]
    colors = [FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR for r in predictions]
    ax2.bar(range(len(predictions)), probs, color=colors, edgecolor='black', linewidth=0.4)
    ax2.set_ylim(0, 105)
    ax2.set_title('신뢰도')
    ax2.grid(axis='y', alpha=0.25)

    ax3 = fig.add_subplot(gs[2])
    grads = [r['max_gradient'] for r in predictions]
    ma_changes = [r['ma_change'] for r in predictions]
    ma_grads = [r['ma_gradient'] for r in predictions]

    ax3.plot(grads, color='purple', alpha=0.85, label='Gradient (최대 변화율)')
    ax3.plot(ma_grads, color='magenta', alpha=0.75, ls='--', label='Moving Avg Gradient')
    ax3.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['gradient'], color='red', ls='--', lw=2, label='Gradient 기준')

    ax3r = ax3.twinx()
    ax3r.plot(ma_changes, color='orange', alpha=0.70, lw=2, label='Moving Avg Change')
    ax3r.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['change_low'], color='brown', ls=':', lw=2, label='Change 기준')

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    ax3.set_title('Gradient / Moving Average')
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[3])
    for i, r in enumerate(predictions):
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        ax4.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='k', lw=1.2))
        txt = '바' if r['label'] == '바닥' else '액'
        txt_color = 'black' if r['label'] == '바닥' else 'white'
        ax4.text(i + 0.5, 0.5, txt, ha='center', va='center', fontweight='bold',
                 fontsize=10 if len(predictions) < 50 else 8, color=txt_color)

    ax4.set_xlim(0, len(predictions))
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_title('예측 시퀀스')
    legend_handles = [
        Patch(facecolor=FLOOR_COLOR, edgecolor='black', label='바닥'),
        Patch(facecolor=LIQUID_COLOR, edgecolor='black', label='액체')
    ]
    ax4.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)

    plt.tight_layout()
    plt.show()

def summarize_predictions(predictions):
    if not predictions:
        print('\n요약할 예측 결과가 없습니다.')
        return
    liquid_labels = [p['label'] for p in predictions if p['label'] != '바닥']
    if not liquid_labels:
        print('\n최종 요약: 액체 이벤트를 확실히 찾지 못했습니다.')
        return
    cnt = Counter(liquid_labels)
    top_label, top_count = cnt.most_common(1)[0]
    print(f'\n최종 요약: 가장 많이 나온 액체 = {top_label} ({top_count} windows)')

def main():
    print('\n' + '=' * 80)
    print('LSTM 액체 감지 예측기 - 로컬 베이스라인 + MA 반영'.center(80))
    print('=' * 80 + '\n')

    while True:
        print('\n' + '─' * 80)
        print(f"지원 바닥: {', '.join(models.keys()) if models else '없음'} | 종료: q")
        print('─' * 80)

        cmd = input('바닥 타입 ▶ ').strip()
        if cmd.lower() in ['q', 'quit', 'exit', 'ㅂㅂ', '']:
            print('\n예측기 종료!')
            break

        floor_type = find_floor_type(cmd)
        if not floor_type:
            print('지원되지 않는 바닥 타입입니다.')
            continue
        if floor_type not in models:
            print(f'해당 바닥 모델이 로드되지 않았습니다: {floor_type}')
            continue

        csv_path = input('CSV 경로 ▶ ').strip().strip('"')
        if not csv_path:
            continue
        if not os.path.exists(csv_path):
            print(f'파일을 찾을 수 없습니다: {csv_path}')
            continue

        try:
            baseline, buffer, changes, times = read_csv_with_fixed_baseline(csv_path)
            if buffer is None or len(buffer) < WINDOW:
                print(f'데이터가 너무 짧습니다. 최소 {WINDOW}개 샘플 필요')
                continue

            print('\n' + '─' * 35)
            print(f'바닥 타입 ▶ {floor_type}')
            print(f'CSV 경로 ▶ "{csv_path}"')

            preds = predict(buffer, changes, baseline, floor_type, models[floor_type])

            for i, p in enumerate(preds[:25], start=1):
                print(
                    f"[{i:02d}] {p['start']:4d}-{p['end']:4d} | {p['label']:>4s} | "
                    f"prob={p['prob']:5.1f} | maxchg={p['max_change']:6.2f} | "
                    f"mchg={p['ma_change']:6.2f} | grad={p['max_gradient']:5.2f} | "
                    f"mgrad={p['ma_gradient']:5.2f} | score={p['event_score']:.2f} | "
                    f"lbase={p['local_baseline']:.2f}"
                )

            summarize_predictions(preds)
            visualize(preds, buffer, changes, baseline, floor_type)

        except Exception as e:
            print(f'예측 중 오류: {e}')

if __name__ == '__main__':
    main()
