import numpy as np
import pickle
import tensorflow as tf
import os
import csv
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from collections import Counter

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = './kumoh_lstm_model_save'
WINDOW = 15
STEP = 5
SMOOTH_K = 5

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

FLOOR_ALIASES = {
    '나무': ['나무', 'wood', '나'],
    '황대': ['황대', '황색대리석', '황색', 'yellow', '황'],
    '회대': ['회대', '회색대리석', '회색', 'gray', '회'],
    '검대': ['검대', '검정색대리석', '검정', 'black', '검'],
    '그마': ['그마', 'greymarble'],
    '207회바': ['회바', '207', '207greyfloor', 'greyfloor'],
    '흰책상': ['흰책상', 'white', 'whitedesk'],
    '나타': ['나타'],
    '회타': ['회타']
}

REVERSE_DIRECTION_FLOORS = {'검대'}


def load_encoder_for_floor(floor_type: str):
    floor_path = os.path.join(SAVE_DIR, f'encoder_{floor_type}.pkl')
    common_path = os.path.join(SAVE_DIR, 'encoder.pkl')
    path = floor_path if os.path.exists(floor_path) else common_path
    if not os.path.exists(path):
        raise FileNotFoundError(f'인코더 파일 없음: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model_for_floor(floor_type: str):
    path = os.path.join(SAVE_DIR, f'model_{floor_type}.keras')
    if not os.path.exists(path):
        raise FileNotFoundError(f'모델 파일 없음: {path}')
    return tf.keras.models.load_model(path)


def find_floor_type(text):
    text = text.lower().strip()
    for floor, aliases in FLOOR_ALIASES.items():
        if text in [a.lower() for a in aliases]:
            return floor
    return None


def _to_float(x):
    try:
        x = str(x).strip()
        return float(x) if x != '' else None
    except Exception:
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
            baseline_raw = _to_float(row.get('baseline_raw', ''))
            base_raw = _to_float(row.get('base_raw', ''))
            current_raw = _to_float(row.get('current_raw', ''))
            raw_data = str(row.get('raw_data', '')).strip()

            if record_type == 'baseline':
                if baseline_raw is not None:
                    baseline_values.append(baseline_raw)
                elif base_raw is not None:
                    baseline_values.append(base_raw)
                continue

            if record_type == 'data' and current_raw is not None:
                data_points.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)
                m = re.search(r'Time\s*=\s*([0-9.]+)s', raw_data)
                timestamps.append(m.group(1) + 's' if m else str(len(data_points) - 1))

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
                    v = _to_float(parts[2])
                    if v is not None:
                        baseline_values.append(v)
                continue

            if len(parts) >= 5 and parts[:4] == ['Final', 'Base', 'Raw', 'Average']:
                v = _to_float(parts[4])
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
                        base_raw = _to_float(val)
                    elif key == 'CurrentRaw':
                        current_raw = _to_float(val)

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


def apply_conservative_postprocess(predictions, floor_type):
    if not predictions:
        return predictions
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    labels = [p['label'] for p in predictions]
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == '바닥':
            i += 1
            continue
        j = i
        while j + 1 < n and labels[j + 1] != '바닥':
            j += 1
        run_len = j - i + 1
        max_change_run = max(predictions[k]['max_change'] for k in range(i, j + 1))
        mean_diff_run = max(predictions[k]['mean_floor_diff'] for k in range(i, j + 1))
        max_noise_run = max(predictions[k].get('noise_ratio', 0.0) for k in range(i, j + 1))
        max_spike_run = max(predictions[k].get('spike_ratio', 0.0) for k in range(i, j + 1))
        if run_len <= 2 and max_change_run < params['change_strong'] + 2 and mean_diff_run < params['plateau_mean'] + 1.0 and (max_noise_run > params.get('noise_ratio', 0.8) or max_spike_run > params.get('spike_ratio', 2.0)):
            for k in range(i, j + 1):
                predictions[k]['label'] = '바닥'
                predictions[k]['prob'] = max(predictions[k]['prob'], 70.0)
        i = j + 1
    return predictions


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


def get_best_non_floor_label(pred, classes):
    if '바닥' in classes:
        floor_idx = np.where(classes == '바닥')[0][0]
        temp = pred.copy()
        temp[floor_idx] = -1.0
    else:
        temp = pred.copy()
    idx = int(np.argmax(temp))
    return classes[idx], float(max(temp[idx] * 100.0, 0.0))


def predict(buffer, changes, baseline, floor_type, model, classes):
    predictions = []
    max_start = len(buffer) - WINDOW
    if max_start < 0:
        return []

    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    clean_buffer = remove_spikes(buffer)
    changes = np.abs(clean_buffer - baseline).astype(np.float32)
    sig_smooth, change_gradient, floor_ref = build_recent_floor_reference(clean_buffer, baseline, floor_type)

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
        max_gradient = float(np.max(np.abs(window_gradient)))
        mean_floor_diff = float(np.mean(np.abs(diff_w)))
        median_floor_diff = float(np.abs(np.median(diff_w)))
        sign_consistency = float(max(np.mean(diff_w >= 0), np.mean(diff_w <= 0)))
        noise_ratio, spike_ratio = compute_noise_metrics(diff_w, smooth_w)

        pred = model.predict(np.transpose(window, (0, 2, 1)), verbose=0)[0]

        orig_idx = int(np.argmax(pred))
        orig_label = classes[orig_idx]
        orig_prob = float(pred[orig_idx] * 100.0)
        best_non_floor_label, best_non_floor_prob = get_best_non_floor_label(pred, classes)

        simple_liquid = max_change >= params['change_low']
        strong_liquid = max_change >= params['change_strong']
        hard_floor = (
            max_change < params['change_low'] * 0.90 and
            mean_floor_diff < params['clear_mean'] and
            max_gradient < params['gradient'] * 0.90
        )
        plateau_flag = (
            mean_floor_diff >= params['plateau_mean'] and
            median_floor_diff >= params['plateau_med'] and
            sign_consistency >= params['sign_keep'] and
            noise_ratio <= params.get('noise_ratio', 0.8) and
            spike_ratio <= params.get('spike_ratio', 2.0)
        )
        very_clear_plateau = (
            mean_floor_diff >= params['plateau_mean'] + 1.8 and
            median_floor_diff >= params['plateau_med'] + 1.2 and
            sign_consistency >= 0.92 and
            noise_ratio <= params.get('noise_ratio', 0.8) * 0.9 and
            spike_ratio <= params.get('spike_ratio', 2.0) * 0.9
        )

        noisy_floor_bias = (noise_ratio > params.get('noise_ratio', 0.8) or spike_ratio > params.get('spike_ratio', 2.0)) and not strong_liquid

        if hard_floor or noisy_floor_bias:
            final_label = '바닥'
            final_prob = 95.0
        elif strong_liquid and orig_label != '바닥' and orig_prob >= 68.0:
            final_label = orig_label
            final_prob = max(orig_prob, 84.0)
        elif strong_liquid and very_clear_plateau:
            final_label = best_non_floor_label
            final_prob = max(best_non_floor_prob, 78.0)
        elif simple_liquid and plateau_flag and orig_label != '바닥' and orig_prob >= 72.0:
            final_label = orig_label
            final_prob = max(orig_prob, 76.0)
        elif simple_liquid and very_clear_plateau and best_non_floor_prob >= 40.0:
            final_label = best_non_floor_label
            final_prob = max(best_non_floor_prob, 70.0)
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
            'mean_floor_diff': mean_floor_diff,
            'median_floor_diff': median_floor_diff,
            'sign_consistency': sign_consistency,
            'noise_ratio': noise_ratio,
            'spike_ratio': spike_ratio,
        })

    return apply_conservative_postprocess(predictions, floor_type)


FLOOR_COLOR = '#e0e0e0'
LIQUID_COLOR = '#7b4a1e'


def visualize(predictions, buffer, changes, baseline, floor_type):
    if not predictions:
        print('시각화할 예측 결과가 없습니다.')
        return

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.0, 2.2, 1.4, 1.8], hspace=0.30)
    fig.suptitle(f'{floor_type} 바닥 - LSTM 실시간 액체 감지 결과', fontsize=20, fontweight='bold', y=0.97)

    ax1 = fig.add_subplot(gs[0])
    for i, r in enumerate(predictions):
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
    ax3.plot(grads, color='purple', alpha=0.85)
    ax3.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['gradient'], color='red', ls='--', lw=2)
    ax3.set_title('Gradient (최대 변화율)')
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[3])
    for i, r in enumerate(predictions):
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        ax4.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='k', lw=1.2))
        txt = '바' if r['label'] == '바닥' else '액'
        ax4.text(i + 0.5, 0.5, txt, ha='center', va='center', fontweight='bold', fontsize=10 if len(predictions) < 50 else 8, color='white')

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


def main():
    print('\n' + '=' * 80)
    print('LSTM 액체 감지 예측기 - refined'.center(80))
    print('=' * 80 + '\n')

    while True:
        print('\n' + '─' * 80)
        print(f"지원 바닥: {', '.join(FLOOR_ALIASES.keys())} | 종료: q")
        print('─' * 80)

        cmd = input('바닥 타입 ▶ ').strip()
        if cmd.lower() in ['q', 'quit', 'exit', 'ㅂㅂ', '']:
            print('\n예측기 종료!')
            break

        floor_type = find_floor_type(cmd)
        if not floor_type:
            print('지원되지 않는 바닥 타입입니다.')
            continue

        csv_path = input('CSV 경로 ▶ ').strip().strip('"')
        if not csv_path:
            continue
        if not os.path.exists(csv_path):
            print(f'파일을 찾을 수 없습니다: {csv_path}')
            continue

        try:
            encoder = load_encoder_for_floor(floor_type)
            classes = encoder.classes_
            model = load_model_for_floor(floor_type)

            baseline, buffer, changes, times = read_csv_with_fixed_baseline(csv_path)
            if buffer is None or len(buffer) < WINDOW:
                print(f'데이터가 너무 짧습니다. 최소 {WINDOW}개 샘플 필요')
                continue

            preds = predict(buffer, changes, baseline, floor_type, model, classes)
            counts = Counter([p['label'] for p in preds])
            liquid_count = sum(v for k, v in counts.items() if k != '바닥')
            print(f"총 윈도우: {len(preds)} | 바닥: {counts.get('바닥', 0)} | 액체: {liquid_count}")
            for label, count in counts.most_common():
                print(f"  {label}: {count}개")
            visualize(preds, buffer, changes, baseline, floor_type)
        except Exception as e:
            print(f'예측 중 오류: {e}')


if __name__ == '__main__':
    main()
