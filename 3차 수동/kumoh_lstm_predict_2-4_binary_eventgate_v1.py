# -*- coding: utf-8 -*-
"""
2-4 구조 기반 이진 예측 코드 v1
- 출력 라벨: 바닥 / 액체
- 2-4의 보수적 로직을 유지하면서, 상태유지 대신 event seed + 짧은 plateau 확장만 허용
- 결과 png와 txt 로그를 같은 폴더에 저장
"""

import os
import csv
import re
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = './kumoh_lstm_model_save'
RESULT_ROOT = './predict_results'
ENCODER_PATH = os.path.join(SAVE_DIR, 'encoder.pkl')

WINDOW = 15
STEP = 5
SMOOTH_K = 5

FLOOR_THRESHOLDS = {
    '나무': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.88, 'alpha': 0.030, 'ref_grad': 0.80, 'ref_band': 6.5, 'noise_ratio': 0.90, 'spike_ratio': 2.40, 'seed_prob': 0.58, 'attach_prob': 0.55, 'expand_max': 2},
    '황대': {'gradient': 0.6, 'change_low': 5.0, 'change_strong': 15.0, 'plateau_mean': 7.0, 'plateau_med': 6.0, 'clear_mean': 2.3, 'sign_keep': 0.88, 'alpha': 0.028, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.82, 'spike_ratio': 2.20, 'seed_prob': 0.57, 'attach_prob': 0.54, 'expand_max': 2},
    '회대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05, 'seed_prob': 0.60, 'attach_prob': 0.56, 'expand_max': 2},
    '검대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05, 'seed_prob': 0.60, 'attach_prob': 0.56, 'expand_max': 2},
    '그마': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05, 'seed_prob': 0.60, 'attach_prob': 0.56, 'expand_max': 2},
    '207회바': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05, 'seed_prob': 0.60, 'attach_prob': 0.56, 'expand_max': 2},
    '흰책상': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'sign_keep': 0.89, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'noise_ratio': 0.78, 'spike_ratio': 2.05, 'seed_prob': 0.60, 'attach_prob': 0.56, 'expand_max': 2},
    '나타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'plateau_mean': 10.0, 'plateau_med': 8.0, 'clear_mean': 3.0, 'sign_keep': 0.90, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0, 'noise_ratio': 0.68, 'spike_ratio': 1.85, 'seed_prob': 0.61, 'attach_prob': 0.57, 'expand_max': 1},
    '회타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'plateau_mean': 10.0, 'plateau_med': 8.0, 'clear_mean': 3.0, 'sign_keep': 0.90, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0, 'noise_ratio': 0.68, 'spike_ratio': 1.85, 'seed_prob': 0.61, 'attach_prob': 0.57, 'expand_max': 1},
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
    '회타': os.path.join(SAVE_DIR, 'model_회타.keras'),
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
        baseline_values, row_bases, data_points, timestamps = [], [], [], []
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
                    timestamps.append(str(len(data_points) - 1))
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
        baseline_values, row_bases, data_points, timestamps = [], [], [], []
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


def create_window(full_signal, start_idx, baseline, floor_type):
    if start_idx + WINDOW > len(full_signal):
        return None
    sig = full_signal[start_idx:start_idx + WINDOW]
    norm_sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    deriv1 = np.diff(norm_sig, prepend=norm_sig[0])
    deriv1 = np.convolve(deriv1, np.ones(3) / 3, mode='same')
    deriv2 = np.diff(deriv1, prepend=deriv1[0])
    base_eff = baseline if len(full_signal) < 20 else 0.7 * baseline + 0.3 * np.mean(full_signal[:20])
    context_start = max(0, start_idx - 5)
    context_end = min(len(full_signal), start_idx + WINDOW + 5)
    recent_mean = np.mean(full_signal[context_start:context_end])
    change_raw = base_eff - recent_mean if floor_type in REVERSE_DIRECTION_FLOORS else recent_mean - base_eff
    direction = np.clip(change_raw / 300.0, -1.0, 1.0)
    direction_channel = np.full((WINDOW,), direction)
    window = np.stack([norm_sig, deriv1, deriv2, direction_channel])
    return window[np.newaxis, :, :]


def safe_load_model(path):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            print(f'모델 로드 실패: {path} | {e}')
    return None


def load_encoder():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f'encoder.pkl 없음: {os.path.abspath(ENCODER_PATH)}')
    encoder = pickle.load(open(ENCODER_PATH, 'rb'))
    return encoder


def compute_window_features(buffer, baseline, floor_type, model, encoder):
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    clean_buffer = remove_spikes(buffer)
    global_changes = np.abs(clean_buffer - baseline).astype(np.float32)
    sig_smooth, change_gradient, floor_ref = build_recent_floor_reference(clean_buffer, baseline, floor_type)

    floor_idx = np.where(encoder.classes_ == '바닥')[0][0]
    liquid_idx = np.where(encoder.classes_ == '액체')[0][0]

    rows = []
    max_start = len(clean_buffer) - WINDOW
    for start_idx in range(0, max_start + 1, STEP):
        window = create_window(clean_buffer, start_idx, baseline, floor_type)
        if window is None:
            continue
        smooth_w = sig_smooth[start_idx:start_idx + WINDOW]
        ref_w = floor_ref[start_idx:start_idx + WINDOW]
        diff_w = smooth_w - ref_w
        chg_w = global_changes[start_idx:start_idx + WINDOW]
        grad_w = change_gradient[start_idx:start_idx + WINDOW]
        abs_grad = np.abs(grad_w)
        flat_std = float(np.std(smooth_w))
        max_change = float(np.max(chg_w))
        avg_change = float(np.mean(chg_w))
        max_gradient = float(np.max(abs_grad))
        edge_ratio = float(np.mean(abs_grad >= params['gradient']))
        mean_floor_diff = float(np.mean(np.abs(diff_w)))
        median_floor_diff = float(np.abs(np.median(diff_w)))
        sign_consistency = float(max(np.mean(diff_w >= 0), np.mean(diff_w <= 0)))
        noise_ratio, spike_ratio = compute_noise_metrics(diff_w, smooth_w)

        pred = model.predict(np.transpose(window, (0, 2, 1)), verbose=0)[0] if model is not None else np.array([1.0, 0.0], dtype=np.float32)
        p_floor = float(pred[floor_idx] * 100.0)
        p_liquid = float(pred[liquid_idx] * 100.0)

        hard_floor = (
            max_change < params['change_low'] * 0.90 and
            mean_floor_diff < params['clear_mean'] and
            max_gradient < params['gradient'] * 0.90
        )
        seed_event = (
            ((max_change >= params['change_strong']) or (max_change >= params['change_low'] and max_gradient >= params['gradient'] * 1.05 and edge_ratio >= 0.18))
            and p_liquid >= params['seed_prob'] * 100.0
        )
        attach_plateau = (
            max_change >= params['change_low'] and
            mean_floor_diff >= params['plateau_mean'] and
            median_floor_diff >= params['plateau_med'] and
            sign_consistency >= params['sign_keep'] and
            noise_ratio <= params.get('noise_ratio', 0.8) * 1.05 and
            spike_ratio <= params.get('spike_ratio', 2.0) * 1.05 and
            p_liquid >= params['attach_prob'] * 100.0
        )
        rows.append({
            'start': start_idx,
            'end': start_idx + WINDOW,
            'p_floor': p_floor,
            'p_liquid': p_liquid,
            'avg_change': avg_change,
            'max_change': max_change,
            'max_gradient': max_gradient,
            'edge_ratio': edge_ratio,
            'mean_floor_diff': mean_floor_diff,
            'median_floor_diff': median_floor_diff,
            'sign_consistency': sign_consistency,
            'noise_ratio': noise_ratio,
            'spike_ratio': spike_ratio,
            'flat_std': flat_std,
            'hard_floor': hard_floor,
            'seed_event': seed_event,
            'attach_plateau': attach_plateau,
            'label': '바닥',
            'prob': max(p_floor, 100 - avg_change),
        })
    return rows, clean_buffer, global_changes


def expand_from_seeds(rows, floor_type):
    if not rows:
        return rows
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    n = len(rows)
    seed_indices = [i for i, r in enumerate(rows) if r['seed_event'] and not r['hard_floor']]

    for i in seed_indices:
        rows[i]['label'] = '액체'
        rows[i]['prob'] = max(rows[i]['p_liquid'], 78.0)

        # left expand
        step_left = 0
        j = i - 1
        while j >= 0 and step_left < params['expand_max']:
            r = rows[j]
            if r['hard_floor']:
                break
            if not r['attach_plateau']:
                break
            rows[j]['label'] = '액체'
            rows[j]['prob'] = max(rows[j]['p_liquid'], 72.0)
            step_left += 1
            j -= 1

        # right expand
        step_right = 0
        j = i + 1
        while j < n and step_right < params['expand_max']:
            r = rows[j]
            if r['hard_floor']:
                break
            if not r['attach_plateau']:
                break
            rows[j]['label'] = '액체'
            rows[j]['prob'] = max(rows[j]['p_liquid'], 72.0)
            step_right += 1
            j += 1

    # bridge single floor gap between liquid runs if the gap itself is attach plateau
    for i in range(1, n - 1):
        if rows[i]['label'] == '바닥' and rows[i-1]['label'] == '액체' and rows[i+1]['label'] == '액체':
            if rows[i]['attach_plateau'] and not rows[i]['hard_floor']:
                rows[i]['label'] = '액체'
                rows[i]['prob'] = max(rows[i]['p_liquid'], 70.0)

    return rows


def conservative_cleanup(rows, floor_type):
    if not rows:
        return rows
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    labels = [r['label'] for r in rows]
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == '바닥':
            i += 1
            continue
        j = i
        while j + 1 < n and labels[j + 1] == '액체':
            j += 1
        run_len = j - i + 1
        has_seed = any(rows[k]['seed_event'] for k in range(i, j + 1))
        max_change_run = max(rows[k]['max_change'] for k in range(i, j + 1))
        max_grad_run = max(rows[k]['max_gradient'] for k in range(i, j + 1))
        if (not has_seed and run_len <= 2) or (not has_seed and max_change_run < params['change_strong'] and max_grad_run < params['gradient'] * 1.10):
            for k in range(i, j + 1):
                rows[k]['label'] = '바닥'
                rows[k]['prob'] = max(rows[k]['p_floor'], 75.0)
        i = j + 1
    return rows


def predict_rows(buffer, baseline, floor_type, model, encoder):
    rows, clean_buffer, changes = compute_window_features(buffer, baseline, floor_type, model, encoder)
    rows = expand_from_seeds(rows, floor_type)
    rows = conservative_cleanup(rows, floor_type)
    return rows, clean_buffer, changes


def make_result_dir(floor_type):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(RESULT_ROOT, f'{floor_type}_{ts}')
    os.makedirs(folder, exist_ok=True)
    return folder


def save_plot(rows, buffer, changes, baseline, floor_type, out_png):
    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.0, 2.2, 1.4, 1.8], hspace=0.30)
    fig.suptitle(f'{floor_type} 바닥 - LSTM 실시간 액체 감지 결과 (2-4 binary)', fontsize=20, fontweight='bold', y=0.97)

    ax1 = fig.add_subplot(gs[0])
    for r in rows:
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
    probs = [r['p_liquid'] if r['label'] == '액체' else r['p_floor'] for r in rows]
    colors = [FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR for r in rows]
    ax2.bar(range(len(rows)), probs, color=colors, edgecolor='black', linewidth=0.4)
    ax2.set_ylim(0, 105)
    ax2.set_title('신뢰도')
    ax2.grid(axis='y', alpha=0.25)

    ax3 = fig.add_subplot(gs[2])
    grads = [r['max_gradient'] for r in rows]
    diffs = [r['mean_floor_diff'] for r in rows]
    ax3.plot(grads, color='purple', alpha=0.85, label='Gradient (최대 변화율)')
    ax3.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['gradient'], color='red', ls='--', lw=2, label='Gradient 기준')
    ax3r = ax3.twinx()
    ax3r.plot(diffs, color='orange', alpha=0.75, label='Mean Floor Diff')
    ax3.legend(loc='upper left', fontsize=10)
    ax3r.legend(loc='upper right', fontsize=10)
    ax3.set_title('Gradient / Floor Diff')
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[3])
    for i, r in enumerate(rows):
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        txt = '바' if r['label'] == '바닥' else '액'
        txt_color = 'black' if r['label'] == '바닥' else 'white'
        ax4.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='k', lw=1.2))
        ax4.text(i + 0.5, 0.5, txt, ha='center', va='center', fontweight='bold', fontsize=8, color=txt_color)
    ax4.set_xlim(0, len(rows))
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_title('예측 시퀀스')
    ax4.legend(handles=[Patch(facecolor=FLOOR_COLOR, edgecolor='black', label='바닥'), Patch(facecolor=LIQUID_COLOR, edgecolor='black', label='액체')], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def save_log(rows, csv_path, floor_type, baseline, out_txt, result_dir):
    cnt = Counter([r['label'] for r in rows])
    total = len(rows)
    liquid_count = cnt.get('액체', 0)
    lines = [
        f'[CSV] {csv_path}',
        f'[결과 폴더] {os.path.abspath(result_dir)}',
        f'[바닥] {floor_type}',
        f'[baseline] {baseline:.3f}',
        f'전체 window 수: {total}',
        f'바닥: {cnt.get("바닥", 0)}개 ({(cnt.get("바닥", 0)/max(total,1))*100:.1f}%)',
        f'액체: {liquid_count}개 ({(liquid_count/max(total,1))*100:.1f}%)',
        '',
        '[상세 window]'
    ]
    for i, r in enumerate(rows, start=1):
        lines.append(
            f'[{i:03d}] {r["start"]:4d}-{r["end"]:4d} | final={r["label"]:>4s} | '
            f'pL={r["p_liquid"]:5.1f} | pF={r["p_floor"]:5.1f} | '
            f'chg={r["max_change"]:6.2f} | grad={r["max_gradient"]:5.2f} | '
            f'edge={r["edge_ratio"]:.2f} | diff={r["mean_floor_diff"]:6.2f} | '
            f'seed={int(r["seed_event"])} | attach={int(r["attach_plateau"])}'
        )
    with open(out_txt, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(lines))


def main():
    print('\n' + '=' * 80)
    print('LSTM 액체 감지 예측기 - 2-4 binary event-gate'.center(80))
    print('=' * 80 + '\n')
    print(f'[현재 작업 폴더] {os.getcwd()}')
    print(f'[모델 폴더] {os.path.abspath(SAVE_DIR)}')
    print(f'[encoder 경로] {os.path.abspath(ENCODER_PATH)}')

    encoder = load_encoder()
    print(f'[encoder classes] {list(encoder.classes_)}')

    models = {}
    for floor, path in FLOOR_MODELS.items():
        print(f'[모델 확인] {os.path.abspath(path)}')
        model = safe_load_model(path)
        if model is not None:
            models[floor] = model
            print(f'  -> 로드 완료: {floor}')
        else:
            print('  -> 파일 없음 또는 로드 실패')

    while True:
        print('\n지원 바닥:', ', '.join(models.keys()))
        print('종료: q')
        cmd = input('바닥 타입 ▶ ').strip()
        if cmd.lower() in ['q', 'quit', 'exit', '']:
            print('종료합니다.')
            break

        floor_type = find_floor_type(cmd)
        if not floor_type or floor_type not in models:
            print('지원되지 않는 바닥 타입입니다.')
            continue

        csv_path = input('CSV 경로 ▶ ').strip().strip('"')
        if not os.path.exists(csv_path):
            print('파일을 찾을 수 없습니다.')
            continue

        baseline, buffer, changes, _ = read_csv_with_fixed_baseline(csv_path)
        if baseline is None:
            print('CSV 파싱 실패')
            continue

        rows, clean_buffer, clean_changes = predict_rows(buffer, baseline, floor_type, models[floor_type], encoder)
        result_dir = make_result_dir(floor_type)
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        out_png = os.path.join(result_dir, f'{stem}_plot.png')
        out_txt = os.path.join(result_dir, f'{stem}_log.txt')
        save_log(rows, csv_path, floor_type, baseline, out_txt, result_dir)
        save_plot(rows, clean_buffer, clean_changes, baseline, floor_type, out_png)

        cnt = Counter([r['label'] for r in rows])
        print(f'\n[결과 폴더] {os.path.abspath(result_dir)}')
        print(f'[로그 저장] {os.path.abspath(out_txt)}')
        print(f'[그림 저장] {os.path.abspath(out_png)}')
        print(f'총 윈도우: {len(rows)} | 바닥: {cnt.get("바닥", 0)} | 액체: {cnt.get("액체", 0)}')


if __name__ == '__main__':
    main()
