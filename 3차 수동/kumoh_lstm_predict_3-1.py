# -*- coding: utf-8 -*-
"""
기존 2-4 구조 유지 + 평탄 구간 억제 강화 버전 (예측)

핵심
- 기존 4채널 유지: norm, deriv1, deriv2, direction_channel
- 기존 WINDOW / STEP 유지
- 하지만 예측 후 바로 window 단위로 칠하지 않고
  "기울기 이벤트"만 액체로 남기도록 후처리 강화
- plateau / flat은 바닥으로 최대한 되돌림
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

EDGE_GRAD_THRESHOLD = 0.85
EDGE_RATIO_THRESHOLD = 0.28
FLAT_STD_THRESHOLD = 0.75
FLAT_GRAD_THRESHOLD = 0.45

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


def create_window(full_signal, start_idx, baseline, floor_type):
    if start_idx + WINDOW > len(full_signal):
        return None
    sig = full_signal[start_idx:start_idx + WINDOW]
    norm_sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    deriv1 = np.diff(norm_sig)
    deriv1 = np.convolve(deriv1, np.ones(3) / 3, 'same')
    deriv1 = np.pad(deriv1, (1, 0))
    deriv2 = np.diff(deriv1)
    deriv2 = np.pad(deriv2, (1, 0))
    base_eff = baseline if len(full_signal) < 20 else 0.7 * baseline + 0.3 * np.mean(full_signal[:20])
    context_start = max(0, start_idx - 5)
    context_end = min(len(full_signal), start_idx + WINDOW + 5)
    recent_mean = np.mean(full_signal[context_start:context_end])
    change_raw = base_eff - recent_mean if floor_type in REVERSE_DIRECTION_FLOORS else recent_mean - base_eff
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


def apply_event_only_postprocess(predictions, floor_type):
    if not predictions:
        return predictions

    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])

    # 1차: flat / plateau 억제
    for p in predictions:
        clear_flat = (
            p['flat_std'] <= FLAT_STD_THRESHOLD
            and p['max_gradient'] <= FLAT_GRAD_THRESHOLD
            and p['mean_floor_diff'] < params['plateau_mean'] * 0.75
        )
        weak_edge = p['edge_ratio'] < EDGE_RATIO_THRESHOLD or p['max_gradient'] < EDGE_GRAD_THRESHOLD

        if clear_flat:
            p['label'] = '바닥'
            p['prob'] = max(p['prob'], 95.0)
            continue

        # plateau인데 edge가 약하면 바닥으로 되돌림
        if p['label'] != '바닥' and weak_edge and p['mean_floor_diff'] >= params['plateau_mean']:
            p['label'] = '바닥'
            p['prob'] = max(p['prob'], 85.0)

    # 2차: 짧은 액체 run 제거
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
        run_edge = max(predictions[k]['edge_ratio'] for k in range(i, j + 1))
        run_grad = max(predictions[k]['max_gradient'] for k in range(i, j + 1))

        if run_len <= 2 or run_edge < EDGE_RATIO_THRESHOLD or run_grad < EDGE_GRAD_THRESHOLD:
            for k in range(i, j + 1):
                predictions[k]['label'] = '바닥'
                predictions[k]['prob'] = max(predictions[k]['prob'], 80.0)
        i = j + 1

    return predictions


def predict(buffer, changes, baseline, floor_type, model):
    predictions = []
    max_start = len(buffer) - WINDOW
    if max_start < 0:
        return []

    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    clean_buffer = remove_spikes(buffer)
    changes = np.abs(clean_buffer - baseline).astype(np.float32)
    sig_smooth, change_gradient, floor_ref = build_recent_floor_reference(clean_buffer, baseline, floor_type)
    floor_idx = np.where(classes == '바닥')[0][0] if '바닥' in classes else -1

    for start_idx in range(0, max_start + 1, STEP):
        window = create_window(clean_buffer, start_idx, baseline, floor_type)
        if window is None:
            continue

        smooth_w = sig_smooth[start_idx:start_idx + WINDOW]
        ref_w = floor_ref[start_idx:start_idx + WINDOW]
        diff_w = smooth_w - ref_w
        window_changes = changes[start_idx:start_idx + WINDOW]
        window_gradient = change_gradient[start_idx:start_idx + WINDOW]

        avg_change = float(np.mean(window_changes))
        max_change = float(np.max(window_changes))
        abs_grad = np.abs(window_gradient)
        max_gradient = float(np.max(abs_grad))
        edge_ratio = float(np.mean(abs_grad >= EDGE_GRAD_THRESHOLD))
        flat_std = float(np.std(smooth_w))

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
        clear_flat_floor = (
            flat_std <= FLAT_STD_THRESHOLD
            and max_gradient <= FLAT_GRAD_THRESHOLD
            and mean_floor_diff < params['clear_mean']
        )
        plateau_flag = (
            mean_floor_diff >= params['plateau_mean']
            and median_floor_diff >= params['plateau_med']
            and sign_consistency >= params['sign_keep']
            and noise_ratio <= params.get('noise_ratio', 0.8)
            and spike_ratio <= params.get('spike_ratio', 2.0)
        )
        edge_event = (
            max_gradient >= EDGE_GRAD_THRESHOLD
            and edge_ratio >= EDGE_RATIO_THRESHOLD
        )

        if clear_flat_floor:
            final_label = '바닥'
            final_prob = 95.0
        elif strong_liquid and edge_event and orig_label != '바닥':
            final_label = orig_label
            final_prob = max(orig_prob, 82.0)
        elif strong_liquid and edge_event:
            final_label = best_non_floor_label
            final_prob = max(best_non_floor_prob, 76.0)
        elif simple_liquid and plateau_flag and not edge_event:
            # plateau는 액체의 유지상태일 수 있지만, 이번 목적은 rise/fall 위주이므로 바닥 처리
            final_label = '바닥'
            final_prob = 85.0
        elif (not simple_liquid) and (not plateau_flag):
            final_label = '바닥'
            final_prob = 92.0
        else:
            final_label = '바닥'
            final_prob = max(orig_prob, 70.0)

        predictions.append({
            'start': start_idx,
            'end': start_idx + WINDOW,
            'label': final_label,
            'prob': final_prob,
            'orig_label': orig_label,
            'orig_prob': orig_prob,
            'max_change': max_change,
            'avg_change': avg_change,
            'max_gradient': max_gradient,
            'edge_ratio': edge_ratio,
            'flat_std': flat_std,
            'mean_floor_diff': mean_floor_diff,
            'median_floor_diff': median_floor_diff,
            'sign_consistency': sign_consistency,
            'noise_ratio': noise_ratio,
            'spike_ratio': spike_ratio,
        })

    predictions = apply_event_only_postprocess(predictions, floor_type)
    return predictions


def visualize(predictions, buffer, changes, baseline, floor_type):
    if not predictions:
        print('시각화할 예측 결과가 없습니다.')
        return

    FLOOR_COLOR = '#E8E8E8'
    LIQUID_COLOR = '#5C2F0F'

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.6, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1_twin = ax1.twinx()
    ax1.plot(buffer, 'b-', lw=2, label='CurrentRaw')
    ax1.axhline(baseline, color='green', ls='--', lw=2, label=f'Base: {baseline:.1f}')
    ax1_twin.plot(changes, 'r:', alpha=0.7, lw=1.8, label='Global Change')

    for r in predictions:
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        alpha = 0.25 if r['label'] == '바닥' else 0.5
        ax1.axvspan(r['start'], min(r['end'], len(buffer)), color=color, alpha=alpha)

    ax1.set_title(f'{floor_type} 바닥 - LSTM 실시간 액체 감지 결과', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    bar_colors = [FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR for r in predictions]
    ax2.bar(range(len(predictions)), [r['prob'] for r in predictions], color=bar_colors,
            alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_ylim(0, 105)
    ax2.set_title('신뢰도')
    ax2.grid(alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[2])
    ax3.plot([r['max_gradient'] for r in predictions], 'purple', alpha=0.8)
    ax3.axhline(EDGE_GRAD_THRESHOLD, color='red', ls='--', lw=2)
    ax3.set_title('Gradient (최대 변화율)')

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
    ax4.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.show()


def summarize_predictions(predictions):
    if not predictions:
        print("\n최종 요약: 예측 결과가 없습니다.")
        return

    cnt = Counter('바닥' if r['label'] == '바닥' else '액체' for r in predictions)
    total = len(predictions)
    print("\n최종 요약:")
    for label, count in cnt.most_common():
        perc = count / total * 100.0
        print(f'  {label}: {count}개 ({perc:.1f}%)')

    liquid_labels = [p['label'] for p in predictions if p['label'] != '바닥']
    if liquid_labels:
        liquid_cnt = Counter(liquid_labels)
        top_label, top_count = liquid_cnt.most_common(1)[0]
        print(f'  가장 많이 나온 액체: {top_label} ({top_count} windows)')
    else:
        print('  액체 이벤트를 확실히 찾지 못했습니다.')


def main():
    print("\n" + "=" * 80)
    print("LSTM 액체 감지 예측기 - 4채널 유지 / 3-1 결과표시 통일".center(80))
    print("=" * 80 + "\n")

    print('사용 가능한 바닥 종류:')
    for floor in models.keys():
        print(f'  - {floor}')
    print()

    floor_type = None
    while floor_type not in models:
        floor_type = find_floor_type(input('바닥 종류를 입력하세요 ▶ '))
        if floor_type not in models:
            print('해당 바닥 모델이 없습니다. 다시 입력하세요.')

    while True:
        path = input('\n예측할 CSV 파일 경로 ▶ ').strip().strip('"')
        if not os.path.exists(path):
            print('파일이 없습니다.')
            continue

        base, buffer, changes, ts = read_csv_with_fixed_baseline(path)
        if base is None:
            print('CSV 파싱 실패')
            continue

        print(f'\n파일: {os.path.basename(path)}')
        print(f'Base: {base:.1f} | 길이: {len(buffer)}')

        predictions = predict(buffer, changes, base, floor_type, models[floor_type])
        print(f'예측 완료: {len(predictions)}개 구간')

        for i, p in enumerate(predictions[:20], start=1):
            print(f"[{i:02d}] {p['start']:4d}-{p['end']:4d} | {p['label']:>4s} | "
                  f"prob={p['prob']:5.1f} | grad={p['max_gradient']:.2f} | "
                  f"edge={p['edge_ratio']:.2f} | flatstd={p['flat_std']:.2f}")

        summarize_predictions(predictions)
        visualize(predictions, buffer, changes, base, floor_type)
        print('시각화 완료!\n')


if __name__ == '__main__':
    main()
