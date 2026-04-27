# -*- coding: utf-8 -*-
import os
import re
import csv
import json
from collections import Counter

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# =========================
# 사용자 설정
# =========================
MODEL_DIR = r"C:\Users\MASL\Desktop\코드개선\3차_리뉴얼\kumoh_binary_model_save"
RESULT_DIR = r"C:\Users\MASL\Desktop\코드개선\3차_리뉴얼\predict_results_binary"
DEFAULT_CSV_PATH = r""

WINDOW = 15
STEP = 5
SMOOTH_K = 5
LOCAL_REF_BACK = 25
LOCAL_REF_GUARD = 3

# 후처리 조정값
ENTRY_SCORE_BONUS = 8.0
EXIT_SCORE_BONUS = 10.0
ENTRY_MIN_RUN = 2
EXIT_MIN_RUN = 2
SHORT_ISLAND_MAX = 1
IGNORE_HEAD_WINDOWS = 2

FLOOR_THRESHOLDS = {
    '나무': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.030, 'ref_grad': 0.80, 'ref_band': 6.5, 'keep_mean_local': 2.0, 'keep_area_local': 30.0, 'force_floor_local': 2.2, 'force_floor_grad': 1.4, 'memory_windows': 2, 'entry_prob': 68.0, 'exit_prob': 42.0, 'return_mean_local': 3.0, 'return_avg_change': 8.0, 'return_floor_diff': 3.0},
    '황대': {'gradient': 0.6, 'change_low': 5.0, 'change_strong': 15.0, 'clear_mean': 2.3, 'alpha': 0.028, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 1.8, 'keep_area_local': 28.0, 'force_floor_local': 2.0, 'force_floor_grad': 1.2, 'memory_windows': 2, 'entry_prob': 68.0, 'exit_prob': 42.0, 'return_mean_local': 2.6, 'return_avg_change': 7.0, 'return_floor_diff': 2.6},
    '회대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 2.0, 'keep_area_local': 32.0, 'force_floor_local': 2.4, 'force_floor_grad': 1.4, 'memory_windows': 2, 'entry_prob': 70.0, 'exit_prob': 44.0, 'return_mean_local': 3.2, 'return_avg_change': 8.0, 'return_floor_diff': 3.0},
    '검대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 2.6, 'keep_area_local': 38.0, 'force_floor_local': 2.5, 'force_floor_grad': 1.5, 'memory_windows': 2, 'entry_prob': 70.0, 'exit_prob': 45.0, 'return_mean_local': 3.4, 'return_avg_change': 8.5, 'return_floor_diff': 3.2},
    '그마': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 2.0, 'keep_area_local': 32.0, 'force_floor_local': 2.4, 'force_floor_grad': 1.4, 'memory_windows': 2, 'entry_prob': 70.0, 'exit_prob': 44.0, 'return_mean_local': 3.2, 'return_avg_change': 8.0, 'return_floor_diff': 3.0},
    '207회바': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 2.0, 'keep_area_local': 32.0, 'force_floor_local': 2.4, 'force_floor_grad': 1.4, 'memory_windows': 2, 'entry_prob': 70.0, 'exit_prob': 44.0, 'return_mean_local': 3.2, 'return_avg_change': 8.0, 'return_floor_diff': 3.0},
    '흰책상': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 1.8, 'keep_area_local': 28.0, 'force_floor_local': 2.2, 'force_floor_grad': 1.3, 'memory_windows': 2, 'entry_prob': 68.0, 'exit_prob': 42.0, 'return_mean_local': 2.8, 'return_avg_change': 7.5, 'return_floor_diff': 2.8},
    '나타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'clear_mean': 3.0, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0, 'keep_mean_local': 1.8, 'keep_area_local': 28.0, 'force_floor_local': 2.2, 'force_floor_grad': 1.3, 'memory_windows': 2, 'entry_prob': 70.0, 'exit_prob': 44.0, 'return_mean_local': 2.8, 'return_avg_change': 8.0, 'return_floor_diff': 2.8},
    '회타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'clear_mean': 3.0, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0, 'keep_mean_local': 2.0, 'keep_area_local': 32.0, 'force_floor_local': 2.4, 'force_floor_grad': 1.4, 'memory_windows': 2, 'entry_prob': 70.0, 'exit_prob': 44.0, 'return_mean_local': 3.0, 'return_avg_change': 8.0, 'return_floor_diff': 3.0},
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
    '회타': ['회타']
}

RAW_RE = re.compile(
    r"Time\s*=?\s*(?P<time>[-+]?\d+(?:\.\d+)?)s?\s*[\|,]?\s*"
    r".*?BaseRaw\s*=?\s*(?P<base>[-+]?\d+(?:\.\d+)?)"
    r".*?CurrentRaw\s*=?\s*(?P<current>[-+]?\d+(?:\.\d+)?)"
    r".*?Change\s*=?\s*(?P<change>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE
)
GRAPH_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def ask_path(prompt, default=""):
    msg = prompt
    if default:
        msg += f" [{default}]"
    msg += " ▶ "
    value = input(msg).strip().strip('"')
    return value if value else default


def find_floor_type(text):
    text = text.lower().strip()
    for floor, aliases in FLOOR_ALIASES.items():
        if text in [a.lower() for a in aliases]:
            return floor
    return None


def _to_float(x):
    try:
        s = str(x).strip()
        if s == '' or s.lower() == 'nan':
            return None
        return float(s)
    except Exception:
        return None


def smooth_signal(x, k=SMOOTH_K):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    k = max(1, min(int(k), len(x)))
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
        if abs(float(sig[i]) - local_med) > k * local_std and abs(float(sig[i]) - float(sig[i - 1])) > 1.2 * local_std and abs(float(sig[i]) - float(sig[i + 1])) > 1.2 * local_std:
            sig[i] = np.float32(0.5 * (sig[i - 1] + sig[i + 1]))
    return sig


def read_csv_with_fixed_baseline(fp):
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

            baseline_values, row_bases, data_points, graph_values = [], [], [], []
            for row in rows:
                record_type = str(row.get('record_type', '')).strip().lower()
                baseline_raw = _to_float(row.get('baseline_raw', ''))
                base_raw = _to_float(row.get('base_raw', ''))
                current_raw = _to_float(row.get('current_raw', ''))
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
                            current_raw = _to_float(m.group('current'))
                            if base_raw is None:
                                base_raw = _to_float(m.group('base'))
                    if current_raw is not None:
                        data_points.append(current_raw)
                    if base_raw is not None:
                        row_bases.append(base_raw)

            if len(data_points) < WINDOW:
                nums = []
                for g in graph_values:
                    nums.extend([float(x) for x in GRAPH_NUM_RE.findall(str(g))])
                if len(nums) >= WINDOW:
                    data_points = nums

            if len(data_points) < WINDOW:
                return None, None, None

            if baseline_values:
                base = float(np.mean(baseline_values))
            elif row_bases:
                base = float(np.mean(row_bases))
            else:
                base = float(np.median(data_points[:min(10, len(data_points))]))

            buffer = np.array(data_points, dtype=np.float32)
            changes = np.abs(buffer - base).astype(np.float32)
            return base, buffer, changes
        except Exception:
            continue
    return None, None, None


def build_recent_floor_reference(signal, baseline, floor_type):
    params = FLOOR_THRESHOLDS[floor_type]
    sig = smooth_signal(remove_spikes(signal))
    grad = np.gradient(sig)
    if len(grad) > 0:
        grad[:IGNORE_HEAD_WINDOWS * STEP] = 0.0
    ref = np.empty_like(sig)
    ref[0] = baseline
    for i in range(1, len(sig)):
        stable_grad = abs(grad[i]) < params['ref_grad']
        close_to_ref = abs(sig[i] - ref[i - 1]) < params['ref_band']
        if stable_grad and close_to_ref:
            ref[i] = (1.0 - params['alpha']) * ref[i - 1] + params['alpha'] * sig[i]
        else:
            ref[i] = ref[i - 1]
    return sig, grad, ref


def compute_moving_reference(signal, start_idx, global_base):
    if start_idx < 10:
        return float(global_base)
    ref_end = max(0, start_idx - LOCAL_REF_GUARD)
    ref_start = max(0, ref_end - LOCAL_REF_BACK)
    ref = signal[ref_start:ref_end]
    if len(ref) < 5:
        return float(global_base)
    ref_smooth = smooth_signal(ref, k=min(5, len(ref)))
    ref_grad = np.abs(np.gradient(ref_smooth)) if len(ref_smooth) >= 2 else np.zeros_like(ref_smooth)
    thr = np.percentile(ref_grad, 70) if len(ref_grad) > 0 else 0.0
    stable_ref = ref_smooth[ref_grad <= thr]
    if len(stable_ref) < 3:
        stable_ref = ref_smooth
    return float(np.mean(stable_ref))


def normalize_1d(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-6:
        sd = 1.0
    return (x - mu) / sd


def pad_or_crop_1d(x, target_len):
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= target_len:
        return x[-target_len:]
    pad_value = x[0] if len(x) > 0 else 0.0
    pad = np.full((target_len - len(x),), pad_value, dtype=np.float32)
    return np.concatenate([pad, x], axis=0)


def make_features(seq):
    seq = np.asarray(seq, dtype=np.float32)
    z = normalize_1d(seq)
    d1 = np.gradient(z).astype(np.float32)
    d2 = np.gradient(d1).astype(np.float32)
    front = max(4, len(seq) // 4)
    base_ref = float(np.mean(seq[:front])) if len(seq) > 0 else 0.0
    diff = normalize_1d(seq - base_ref)
    return np.stack([z, d1, d2, diff], axis=-1).astype(np.float32)


def create_model_input(full_signal, start_idx, target_len):
    seq = full_signal[start_idx:start_idx + WINDOW]
    if len(seq) == 0:
        return None
    seq = pad_or_crop_1d(seq, target_len)
    feat = make_features(seq)
    return np.expand_dims(feat, axis=0)


def compute_scores(p, params):
    liquid_score = 0.0
    liquid_score += max(0.0, p['liquid_prob'] - params['entry_prob']) * 0.85
    liquid_score += max(0.0, p['max_change'] - params['change_low']) * 1.40
    liquid_score += max(0.0, p['mean_local_change'] - params['keep_mean_local']) * 7.50
    liquid_score += max(0.0, p['mean_floor_diff'] - params['clear_mean']) * 5.50
    liquid_score += max(0.0, p['max_gradient'] - params['gradient']) * 2.20
    if p['max_change'] >= params['change_strong']:
        liquid_score += ENTRY_SCORE_BONUS

    floor_score = 0.0
    floor_score += max(0.0, params['exit_prob'] - p['liquid_prob']) * 0.95
    floor_score += max(0.0, params['return_mean_local'] - p['mean_local_change']) * 11.0
    floor_score += max(0.0, params['return_avg_change'] - p['avg_change']) * 2.40
    floor_score += max(0.0, params['return_floor_diff'] - p['mean_floor_diff']) * 8.50
    floor_score += max(0.0, params['force_floor_grad'] - p['max_gradient']) * 3.20
    if p['mean_local_change'] <= params['force_floor_local'] and p['max_gradient'] <= params['force_floor_grad']:
        floor_score += EXIT_SCORE_BONUS
    if p['avg_change'] < params['change_low'] and p['mean_floor_diff'] < params['clear_mean']:
        floor_score += EXIT_SCORE_BONUS * 0.7

    return liquid_score, floor_score


def initial_decision(p, params, threshold):
    force_floor = p['mean_local_change'] <= params['force_floor_local'] and p['max_gradient'] <= params['force_floor_grad']
    strong_liquid = p['max_change'] >= params['change_strong'] and p['mean_local_change'] >= params['keep_mean_local']
    weak_floor = p['avg_change'] < params['change_low'] and p['mean_floor_diff'] < params['clear_mean']

    if force_floor:
        final_label = '바닥'
    elif strong_liquid:
        final_label = '액체'
    elif weak_floor and p['liquid_prob'] < max(60.0, threshold * 100.0 + 5.0):
        final_label = '바닥'
    else:
        liquid_score, floor_score = compute_scores(p, params)
        final_label = '액체' if liquid_score >= floor_score else '바닥'

    liquid_score, floor_score = compute_scores(p, params)
    gap = abs(liquid_score - floor_score)
    if final_label == '액체':
        final_prob = max(float(p['liquid_prob']), min(96.0, 60.0 + gap))
    else:
        final_prob = max(float(p['floor_prob']), min(96.0, 60.0 + gap))
    return final_label, final_prob, liquid_score, floor_score


def fill_short_islands(labels, max_len=1):
    if len(labels) < 3:
        return labels[:]
    out = labels[:]
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        left = out[i - 1] if i - 1 >= 0 else None
        right = out[j] if j < n else None
        if run_len <= max_len and left is not None and right is not None and left == right:
            for k in range(i, j):
                out[k] = left
        i = j
    return out


def enforce_min_runs(labels, target_label, min_run):
    out = labels[:]
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if out[i] == target_label and run_len < min_run:
            left = out[i - 1] if i - 1 >= 0 else None
            right = out[j] if j < n else None
            repl = right if left is None else left
            if left is not None and right is not None and left == right:
                repl = left
            elif left is None and right is None:
                repl = '바닥' if target_label == '액체' else '액체'
            for k in range(i, j):
                out[k] = repl
        i = j
    return out


def sequential_postprocess(predictions, floor_type):
    if not predictions:
        return predictions
    params = FLOOR_THRESHOLDS[floor_type]
    labels = [p['label'] for p in predictions]

    labels = fill_short_islands(labels, max_len=SHORT_ISLAND_MAX)
    labels = enforce_min_runs(labels, '액체', ENTRY_MIN_RUN)
    labels = enforce_min_runs(labels, '바닥', EXIT_MIN_RUN)

    out = labels[:]
    state = out[0]
    liquid_streak = 0
    floor_streak = 0
    for i, p in enumerate(predictions):
        liquid_score, floor_score = compute_scores(p, params)
        clear_floor = floor_score >= liquid_score + 6.0
        clear_liquid = liquid_score >= floor_score + 4.0

        if i == 0:
            state = out[i]
            continue

        if state == '바닥':
            if clear_liquid:
                liquid_streak += 1
            else:
                liquid_streak = 0
            if liquid_streak >= ENTRY_MIN_RUN:
                state = '액체'
                liquid_streak = 0
        else:
            if clear_floor:
                floor_streak += 1
            else:
                floor_streak = 0
            if floor_streak >= EXIT_MIN_RUN:
                state = '바닥'
                floor_streak = 0
        out[i] = state

    for i, lbl in enumerate(out):
        predictions[i]['label'] = lbl
        if lbl == '액체':
            predictions[i]['prob'] = max(predictions[i]['prob'], predictions[i]['liquid_prob'])
        else:
            predictions[i]['prob'] = max(predictions[i]['prob'], predictions[i]['floor_prob'])
    return predictions


def predict(buffer, changes, baseline, floor_type, model, cfg):
    predictions = []
    max_start = len(buffer) - WINDOW
    if max_start < 0:
        return []

    threshold = float(cfg.get('threshold', 0.5))
    target_len = int(cfg['target_len'])
    params = FLOOR_THRESHOLDS[floor_type]

    clean_buffer = remove_spikes(buffer)
    changes = np.abs(clean_buffer - baseline).astype(np.float32)
    sig_smooth, change_gradient, floor_ref = build_recent_floor_reference(clean_buffer, baseline, floor_type)

    for wi, start_idx in enumerate(range(0, max_start + 1, STEP)):
        end_idx = start_idx + WINDOW
        model_input = create_model_input(clean_buffer, start_idx, target_len)
        if model_input is None:
            continue

        smooth_w = sig_smooth[start_idx:end_idx]
        ref_w = floor_ref[start_idx:end_idx]
        diff_w = smooth_w - ref_w
        window_changes = changes[start_idx:end_idx]
        window_gradient = change_gradient[start_idx:end_idx].copy()
        if wi < IGNORE_HEAD_WINDOWS and len(window_gradient) > 0:
            window_gradient[:] = 0.0

        moving_ref = compute_moving_reference(sig_smooth, start_idx, baseline)
        local_abs = np.abs(smooth_w - moving_ref)
        mean_local_change = float(np.mean(local_abs))
        area_local_change = float(np.sum(local_abs))
        avg_change = float(np.mean(window_changes))
        max_change = float(np.max(window_changes))
        max_gradient = float(np.max(np.abs(window_gradient))) if len(window_gradient) else 0.0
        mean_floor_diff = float(np.mean(np.abs(diff_w)))

        liquid_prob = float(model.predict(model_input, verbose=0)[0][0] * 100.0)
        floor_prob = 100.0 - liquid_prob

        temp = {
            'start': start_idx,
            'end': end_idx,
            'floor_prob': floor_prob,
            'liquid_prob': liquid_prob,
            'max_change': max_change,
            'avg_change': avg_change,
            'mean_local_change': mean_local_change,
            'area_local_change': area_local_change,
            'max_gradient': max_gradient,
            'mean_floor_diff': mean_floor_diff,
        }
        final_label, final_prob, liquid_score, floor_score = initial_decision(temp, params, threshold)
        temp['label'] = final_label
        temp['prob'] = final_prob
        temp['liquid_score'] = liquid_score
        temp['floor_score'] = floor_score
        predictions.append(temp)

    return sequential_postprocess(predictions, floor_type)


def summarize_predictions(predictions):
    cnt = Counter([p['label'] for p in predictions])
    print(f"\n최종 요약: 바닥 {cnt.get('바닥', 0)} windows / 액체 {cnt.get('액체', 0)} windows")


def save_outputs(predictions, buffer, changes, baseline, floor_type, src_csv_path, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    src_name = os.path.splitext(os.path.basename(src_csv_path))[0]
    out_csv = os.path.join(result_dir, f"{src_name}_{floor_type}_binary_predictions.csv")
    out_png = os.path.join(result_dir, f"{src_name}_{floor_type}_binary_plot.png")

    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['window_index', 'start', 'end', 'label', 'prob', 'floor_prob', 'liquid_prob', 'max_change', 'avg_change', 'mean_local_change', 'area_local_change', 'max_gradient', 'mean_floor_diff', 'liquid_score', 'floor_score'])
        for i, p in enumerate(predictions):
            writer.writerow([i, p['start'], p['end'], p['label'], round(float(p['prob']), 4), round(float(p['floor_prob']), 4), round(float(p['liquid_prob']), 4), round(float(p['max_change']), 4), round(float(p['avg_change']), 4), round(float(p['mean_local_change']), 4), round(float(p['area_local_change']), 4), round(float(p['max_gradient']), 4), round(float(p['mean_floor_diff']), 4), round(float(p.get('liquid_score', 0.0)), 4), round(float(p.get('floor_score', 0.0)), 4)])

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
    ax1.set_title(f'{floor_type} 바닥 - LSTM 실시간 액체 감지 결과', fontsize=22, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    bar_colors = [FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR for r in predictions]
    ax2.bar(range(len(predictions)), [r['prob'] for r in predictions], color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_ylim(0, 105)
    ax2.set_title('신뢰도', fontsize=18)
    ax2.grid(alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[2])
    ax3.plot([r['max_gradient'] for r in predictions], color='purple', alpha=0.8, label='Gradient')
    ax3.plot([r['mean_local_change'] for r in predictions], color='orange', alpha=0.8, label='Moving Baseline Change')
    ax3.axhline(FLOOR_THRESHOLDS[floor_type]['gradient'], color='red', ls='--', lw=2, label='Gradient 기준')
    ax3.legend(loc='upper right')
    ax3.set_title('Gradient / Moving Baseline Change', fontsize=16)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[3])
    for i, r in enumerate(predictions):
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        ax4.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', lw=1.2))
        text_color = 'black' if r['label'] == '바닥' else 'white'
        ax4.text(i + 0.5, 0.5, '바' if r['label'] == '바닥' else '액', ha='center', va='center', fontsize=9, color=text_color, fontweight='bold')
    ax4.set_xlim(0, len(predictions))
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_title('예측 시퀀스', fontsize=18)
    legend_handles = [Patch(facecolor=FLOOR_COLOR, edgecolor='black', label='바닥'), Patch(facecolor=LIQUID_COLOR, edgecolor='black', label='액체')]
    ax4.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print('예측 CSV 저장:', out_csv)
    print('예측 그림 저장:', out_png)


def main():
    print(f'[MODEL_DIR]  {MODEL_DIR}')
    print(f'[RESULT_DIR] {RESULT_DIR}')

    model_dir = ask_path('모델 폴더', MODEL_DIR)
    result_dir = ask_path('결과 저장 폴더', RESULT_DIR)

    model_path = os.path.join(model_dir, 'binary_liquid_floor.keras')
    config_path = os.path.join(model_dir, 'binary_liquid_floor_config.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'모델 파일 없음: {model_path}')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'설정 파일 없음: {config_path}')

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model = tf.keras.models.load_model(model_path)

    while True:
        print('\n지원 바닥:', ', '.join(FLOOR_THRESHOLDS.keys()))
        print('종료: q')
        cmd = input('바닥 타입 ▶ ').strip()
        if cmd.lower() in ['q', 'quit', 'exit', '']:
            break

        floor_type = find_floor_type(cmd)
        if not floor_type:
            print('지원되지 않는 바닥 타입입니다.')
            continue

        csv_path = ask_path('예측할 CSV 전체 경로', DEFAULT_CSV_PATH)
        if not csv_path or not os.path.exists(csv_path):
            print('파일을 찾을 수 없습니다.')
            continue

        baseline, buffer, changes = read_csv_with_fixed_baseline(csv_path)
        if baseline is None:
            print('CSV를 읽지 못했습니다. 최소 길이 또는 컬럼 구성을 확인하세요.')
            continue

        preds = predict(buffer, changes, baseline, floor_type, model, cfg)
        if not preds:
            print('예측 window가 생성되지 않았습니다.')
            continue

        for i, p in enumerate(preds[:25], start=1):
            print(f"[{i:02d}] {p['start']:4d}-{p['end']:4d} | {p['label']:>4s} | prob={p['prob']:5.1f} | floor={p['floor_prob']:5.1f} | liquid={p['liquid_prob']:5.1f} | maxchg={p['max_change']:6.2f} | mbase={p['mean_local_change']:5.2f} | grad={p['max_gradient']:5.2f}")
        summarize_predictions(preds)
        save_outputs(preds, buffer, changes, baseline, floor_type, csv_path, result_dir)


if __name__ == '__main__':
    main()
