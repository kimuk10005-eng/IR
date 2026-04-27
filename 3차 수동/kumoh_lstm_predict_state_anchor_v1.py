# -*- coding: utf-8 -*-
"""
3차 상태유지 실험용 예측 코드 v1
핵심
- 4채널 유지: norm, deriv1, deriv2, direction_channel
- rise/fall 이벤트가 나오면 직전 바닥 평균을 anchor baseline으로 고정
- 액체 plateau는 평평해도 anchor baseline 대비 offset이 크면 LIQUID_HOLD 유지
- 결과 png와 txt 로그를 같은 폴더(results_타임스탬프)에 저장
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
    '황대': {'grad_enter': 0.65, 'change_enter': 5.5, 'hold_offset': 6.0, 'hold_release': 3.2, 'flat_std': 1.00, 'release_need': 2},
    '회대': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'hold_release': 3.8, 'flat_std': 1.10, 'release_need': 2},
    '나타': {'grad_enter': 0.85, 'change_enter': 6.5, 'hold_offset': 7.5, 'hold_release': 4.2, 'flat_std': 1.20, 'release_need': 2},
    '회타': {'grad_enter': 0.85, 'change_enter': 6.5, 'hold_offset': 7.5, 'hold_release': 4.2, 'flat_std': 1.20, 'release_need': 2},
    '검대': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'hold_release': 3.8, 'flat_std': 1.10, 'release_need': 2},
    '그마': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'hold_release': 3.8, 'flat_std': 1.10, 'release_need': 2},
    '207회바': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'hold_release': 3.8, 'flat_std': 1.10, 'release_need': 2},
    '흰책상': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'hold_release': 3.8, 'flat_std': 1.10, 'release_need': 2},
    '나무': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 7.0, 'hold_release': 3.8, 'flat_std': 1.10, 'release_need': 2},
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


def read_csv_with_fixed_baseline(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    def parse_new(lines):
        baseline_values, row_bases, data_points = [], [], []
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
                    data_points.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)
        if len(data_points) < WINDOW:
            return None, None, None
        if baseline_values:
            base = float(np.mean(baseline_values))
        elif row_bases:
            base = float(np.mean(row_bases))
        else:
            return None, None, None
        buffer = np.array(data_points, dtype=np.float32)
        changes = np.abs(buffer - base).astype(np.float32)
        return base, buffer, changes

    def parse_old(lines):
        baseline_values, row_bases, data_points = [], [], []
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
                    data_points.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)
        if len(data_points) < WINDOW:
            return None, None, None
        if final_base is not None:
            base = float(final_base)
        elif baseline_values:
            base = float(np.mean(baseline_values))
        elif row_bases:
            base = float(np.mean(row_bases))
        else:
            return None, None, None
        buffer = np.array(data_points, dtype=np.float32)
        changes = np.abs(buffer - base).astype(np.float32)
        return base, buffer, changes

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


def create_window(signal, start_idx, baseline, floor_type, anchor_base=None):
    if start_idx + WINDOW > len(signal):
        return None
    sig = signal[start_idx:start_idx + WINDOW]
    norm_sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    deriv1 = np.diff(norm_sig, prepend=norm_sig[0])
    deriv1 = np.convolve(deriv1, np.ones(3) / 3, 'same')
    deriv2 = np.diff(deriv1, prepend=deriv1[0])
    if anchor_base is None:
        anchor_base = baseline
    context_start = max(0, start_idx - 5)
    context_end = min(len(signal), start_idx + WINDOW + 5)
    recent_mean = np.mean(signal[context_start:context_end])
    change_raw = anchor_base - recent_mean if floor_type in REVERSE_DIRECTION_FLOORS else recent_mean - anchor_base
    direction = np.clip(change_raw / 300.0, -1.0, 1.0)
    direction_channel = np.full((WINDOW,), direction, dtype=np.float32)
    window = np.stack([norm_sig, deriv1, deriv2, direction_channel], axis=0)
    return window[np.newaxis, :, :]


def load_models_and_encoder():
    print(f'[모델 폴더] {os.path.abspath(SAVE_DIR)}')
    print(f'[encoder 경로] {os.path.abspath(ENCODER_PATH)}')
    encoder = pickle.load(open(ENCODER_PATH, 'rb'))
    classes = encoder.classes_
    models = {}
    for floor, path in FLOOR_MODELS.items():
        print(f'[모델 확인] {os.path.abspath(path)}')
        if os.path.exists(path):
            try:
                models[floor] = tf.keras.models.load_model(path)
                print(f'  -> 로드 완료: {floor}')
            except Exception as e:
                print(f'  -> 로드 실패: {e}')
        else:
            print('  -> 파일 없음')
    return encoder, classes, models


def get_best_non_floor_label(pred, classes):
    floor_idx = np.where(classes == '바닥')[0][0] if '바닥' in classes else -1
    temp = pred.copy()
    if floor_idx >= 0:
        temp[floor_idx] = -1.0
    idx = int(np.argmax(temp))
    return str(classes[idx]), float(max(temp[idx] * 100.0, 0.0))


def compute_anchor_from_history(signal, start_idx, fallback):
    left = max(0, start_idx - 20)
    right = max(0, start_idx - 1)
    if right - left + 1 >= 6:
        hist = signal[left:right + 1]
        return float(np.mean(hist))
    return float(fallback)


def fill_small_floor_gaps(predictions, max_gap=1):
    labels = [p['label'] for p in predictions]
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] != '바닥':
            i += 1
            continue
        j = i
        while j + 1 < n and labels[j + 1] == '바닥':
            j += 1
        gap_len = j - i + 1
        prev_liquid = i - 1 >= 0 and labels[i - 1] != '바닥'
        next_liquid = j + 1 < n and labels[j + 1] != '바닥'
        if gap_len <= max_gap and prev_liquid and next_liquid:
            fill_label = labels[i - 1]
            for k in range(i, j + 1):
                predictions[k]['label'] = fill_label
                predictions[k]['state'] = 'LIQUID_HOLD'
        i = j + 1
    return predictions


def remove_short_liquid_runs(predictions, min_run=2):
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
        if run_len < min_run:
            for k in range(i, j + 1):
                predictions[k]['label'] = '바닥'
                predictions[k]['state'] = 'FLOOR'
        i = j + 1
    return predictions


def predict(buffer, baseline, floor_type, model, classes):
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    signal = smooth_signal(remove_spikes(buffer))
    global_change = np.abs(signal - baseline)
    global_grad = np.gradient(signal)

    predictions = []
    state = 'FLOOR'
    anchor_baseline = float(baseline)
    hold_label = '액체'
    release_counter = 0

    floor_idx = np.where(classes == '바닥')[0][0] if '바닥' in classes else -1

    for start_idx in range(0, len(signal) - WINDOW + 1, STEP):
        win = signal[start_idx:start_idx + WINDOW]
        grad_w = global_grad[start_idx:start_idx + WINDOW]
        max_grad = float(np.max(np.abs(grad_w)))
        mean_offset_global = float(np.mean(global_change[start_idx:start_idx + WINDOW]))
        std_w = float(np.std(win))

        anchor_offset = np.abs(win - anchor_baseline)
        mean_anchor_offset = float(np.mean(anchor_offset))
        max_anchor_offset = float(np.max(anchor_offset))
        edge_ratio = float(np.mean(np.abs(grad_w) >= params['grad_enter']))
        flat_flag = (std_w <= params['flat_std'] and max_grad < params['grad_enter'] * 0.95)
        enter_event = (
            max_grad >= params['grad_enter']
            and max_anchor_offset >= params['change_enter']
            and edge_ratio >= 0.20
        )
        hold_cond = (
            mean_anchor_offset >= params['hold_offset']
            or (flat_flag and mean_anchor_offset >= params['hold_offset'] * 0.85)
        )
        release_cond = (
            mean_anchor_offset <= params['hold_release']
            and max_grad < params['grad_enter'] * 1.10
        )

        window = create_window(signal, start_idx, baseline, floor_type, anchor_baseline)
        if model is not None:
            pred = model.predict(np.transpose(window, (0, 2, 1)), verbose=0)[0]
        else:
            pred = np.zeros(len(classes), dtype=np.float32)
            pred[floor_idx] = 1.0
        orig_idx = int(np.argmax(pred))
        orig_label = str(classes[orig_idx])
        orig_prob = float(pred[orig_idx] * 100.0)
        best_non_floor_label, best_non_floor_prob = get_best_non_floor_label(pred, classes)

        if state == 'FLOOR':
            if enter_event:
                anchor_baseline = compute_anchor_from_history(signal, start_idx, baseline)
                anchor_offset = np.abs(win - anchor_baseline)
                mean_anchor_offset = float(np.mean(anchor_offset))
                hold_label = best_non_floor_label if best_non_floor_label != '바닥' else '액체'
                state = 'LIQUID_HOLD'
                release_counter = 0
                final_label = hold_label
            else:
                anchor_baseline = 0.97 * anchor_baseline + 0.03 * float(np.mean(win))
                final_label = '바닥'
        else:  # LIQUID_HOLD
            if hold_cond:
                final_label = hold_label
                release_counter = 0
            elif release_cond:
                release_counter += 1
                if release_counter >= params['release_need']:
                    state = 'FLOOR'
                    anchor_baseline = compute_anchor_from_history(signal, start_idx, baseline)
                    final_label = '바닥'
                    release_counter = 0
                else:
                    final_label = hold_label
            elif enter_event:
                final_label = hold_label
                release_counter = 0
            else:
                final_label = hold_label if mean_anchor_offset >= params['hold_release'] * 1.15 else '바닥'
                if final_label == '바닥':
                    state = 'FLOOR'
                    release_counter = 0

        predictions.append({
            'start': start_idx,
            'end': start_idx + WINDOW,
            'label': final_label,
            'state': state,
            'orig_label': orig_label,
            'orig_prob': orig_prob,
            'best_non_floor_label': best_non_floor_label,
            'best_non_floor_prob': best_non_floor_prob,
            'max_grad': max_grad,
            'edge_ratio': edge_ratio,
            'std': std_w,
            'mean_anchor_offset': mean_anchor_offset,
            'max_anchor_offset': max_anchor_offset,
            'mean_global_change': mean_offset_global,
            'anchor_baseline': anchor_baseline,
        })

    predictions = fill_small_floor_gaps(predictions, max_gap=1)
    predictions = remove_short_liquid_runs(predictions, min_run=2)
    return predictions, global_change


def summarize_predictions(predictions):
    total = len(predictions)
    liquid = sum(1 for p in predictions if p['label'] != '바닥')
    floor = total - liquid
    out = []
    out.append(f'전체 window 수: {total}')
    out.append(f'바닥: {floor}개 ({floor / max(total,1) * 100:.1f}%)')
    out.append(f'액체: {liquid}개 ({liquid / max(total,1) * 100:.1f}%)')
    liquid_labels = [p['label'] for p in predictions if p['label'] != '바닥']
    if liquid_labels:
        c = Counter(liquid_labels)
        top_label, top_count = c.most_common(1)[0]
        out.append(f'가장 많이 나온 액체: {top_label} ({top_count} windows)')
    else:
        out.append('가장 많이 나온 액체: 없음')
    return out


def save_outputs(predictions, buffer, global_change, baseline, floor_type, csv_path, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    log_path = os.path.join(result_dir, f'{base_name}_log.txt')
    img_path = os.path.join(result_dir, f'{base_name}_plot.png')

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'[CSV] {os.path.abspath(csv_path)}\n')
        f.write(f'[결과 폴더] {os.path.abspath(result_dir)}\n')
        f.write(f'[바닥] {floor_type}\n')
        f.write(f'[baseline] {baseline:.3f}\n')
        for line in summarize_predictions(predictions):
            f.write(line + '\n')
        f.write('\n[상세 window]\n')
        for i, p in enumerate(predictions, start=1):
            f.write(
                f"[{i:03d}] {p['start']:4d}-{p['end']:4d} | final={p['label']:>4s} | state={p['state']:>11s} | "
                f"model={p['orig_label']:>4s}({p['orig_prob']:5.1f}) | nonfloor={p['best_non_floor_label']:>4s}({p['best_non_floor_prob']:5.1f}) | "
                f"anch_ofs={p['mean_anchor_offset']:6.2f}/{p['max_anchor_offset']:6.2f} | "
                f"chg={p['mean_global_change']:6.2f} | grad={p['max_grad']:5.2f} | edge={p['edge_ratio']:.2f} | "
                f"anchor={p['anchor_baseline']:.2f}\n"
            )

    FLOOR_COLOR = '#E8E8E8'
    LIQUID_COLOR = '#7B4A1E'

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.5, 1.5, 1.2, 1.3], hspace=0.28)
    fig.suptitle(f'{floor_type} 바닥 - 상태유지 기반 액체 감지 결과', fontsize=20, fontweight='bold', y=0.97)

    ax1 = fig.add_subplot(gs[0])
    ax1r = ax1.twinx()
    for p in predictions:
        color = FLOOR_COLOR if p['label'] == '바닥' else LIQUID_COLOR
        ax1.axvspan(p['start'], p['end'], color=color, alpha=0.25 if p['label'] == '바닥' else 0.35)
    ax1.plot(buffer, color='blue', lw=2, label='CurrentRaw')
    ax1.axhline(baseline, color='green', ls='--', lw=2, label=f'Base: {baseline:.1f}')
    ax1r.plot(global_change, color='red', ls=':', lw=2, alpha=0.75, label='Global Change')
    ax1.legend(loc='upper left')
    ax1r.legend(loc='upper right')
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[1])
    probs = [max(p['orig_prob'], p['best_non_floor_prob']) if p['label'] != '바닥' else max(70.0, 100.0 - p['mean_anchor_offset']) for p in predictions]
    colors = [FLOOR_COLOR if p['label'] == '바닥' else LIQUID_COLOR for p in predictions]
    ax2.bar(range(len(predictions)), probs, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylim(0, 105)
    ax2.set_title('신뢰도')
    ax2.grid(axis='y', alpha=0.25)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot([p['max_grad'] for p in predictions], color='purple', label='Gradient (최대 변화율)')
    ax3.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['grad_enter'], color='red', ls='--', lw=2, label='Gradient 기준')
    ax3r = ax3.twinx()
    ax3r.plot([p['mean_anchor_offset'] for p in predictions], color='orange', lw=2, alpha=0.80, label='Anchor Offset')
    ax3r.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['hold_offset'], color='brown', ls=':', lw=2, label='Hold 기준')
    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3r.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, loc='upper right')
    ax3.set_title('Gradient / Anchor Offset')
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[3])
    for i, p in enumerate(predictions):
        color = FLOOR_COLOR if p['label'] == '바닥' else LIQUID_COLOR
        ax4.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='k', lw=1.1))
        txt = '바' if p['label'] == '바닥' else '액'
        txt_color = 'black' if p['label'] == '바닥' else 'white'
        ax4.text(i + 0.5, 0.5, txt, ha='center', va='center', fontsize=8, fontweight='bold', color=txt_color)
    ax4.set_xlim(0, len(predictions))
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_title('예측 시퀀스')
    ax4.legend(handles=[Patch(facecolor=FLOOR_COLOR, edgecolor='black', label='바닥'), Patch(facecolor=LIQUID_COLOR, edgecolor='black', label='액체')],
               loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.savefig(img_path, dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return log_path, img_path


def main():
    print('\n' + '=' * 80)
    print('LSTM 액체 감지 예측기 - 상태유지(anchor baseline) 실험'.center(80))
    print('=' * 80 + '\n')

    encoder, classes, models = load_models_and_encoder()
    if not models:
        print('로드된 모델이 없습니다.')
        return

    while True:
        print('\n지원 바닥:', ', '.join(models.keys()))
        print('종료: q')
        cmd = input('바닥 타입 ▶ ').strip()
        if cmd.lower() in ['q', 'quit', 'exit', '']:
            print('종료합니다.')
            break
        floor_type = find_floor_type(cmd)
        if floor_type not in models:
            print('지원되지 않거나 로드되지 않은 바닥 타입입니다.')
            continue

        csv_path = input('CSV 경로 ▶ ').strip().strip('"')
        if not os.path.exists(csv_path):
            print('파일을 찾을 수 없습니다.')
            continue

        baseline, buffer, _changes = read_csv_with_fixed_baseline(csv_path)
        if baseline is None:
            print('CSV 파싱 실패')
            continue

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(RESULT_ROOT, f'{floor_type}_{stamp}')
        print(f'[결과 저장 폴더] {os.path.abspath(result_dir)}')

        predictions, global_change = predict(buffer, baseline, floor_type, models[floor_type], classes)
        print(f'[예측 대상] floor={floor_type} | samples={len(buffer)} | baseline={baseline:.3f}')
        for i, p in enumerate(predictions[:25], start=1):
            print(
                f"[{i:02d}] {p['start']:4d}-{p['end']:4d} | final={p['label']:>4s} | state={p['state']:>11s} | "
                f"model={p['orig_label']:>4s}({p['orig_prob']:5.1f}) | anch={p['mean_anchor_offset']:5.2f}/{p['max_anchor_offset']:5.2f} | "
                f"chg={p['mean_global_change']:5.2f} | grad={p['max_grad']:5.2f} | edge={p['edge_ratio']:.2f}"
            )
        for line in summarize_predictions(predictions):
            print(line)
        log_path, img_path = save_outputs(predictions, buffer, global_change, baseline, floor_type, csv_path, result_dir)
        print(f'[로그 저장] {os.path.abspath(log_path)}')
        print(f'[이미지 저장] {os.path.abspath(img_path)}')


if __name__ == '__main__':
    main()
