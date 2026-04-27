# -*- coding: utf-8 -*-
import os
import csv
import re
import pickle
from collections import Counter

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

WINDOW = 15
STEP = 5
SMOOTH_K = 5
LOCAL_REF_BACK = 25
LOCAL_REF_GUARD = 3

FLOOR_THRESHOLDS = {
    '나무': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.030, 'ref_grad': 0.80, 'ref_band': 6.5, 'keep_mean_local': 2.0, 'keep_area_local': 30.0, 'force_floor_local': 2.2, 'force_floor_grad': 1.4, 'memory_windows': 2},
    '황대': {'gradient': 0.6, 'change_low': 5.0, 'change_strong': 15.0, 'clear_mean': 2.3, 'alpha': 0.028, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 1.8, 'keep_area_local': 28.0, 'force_floor_local': 2.0, 'force_floor_grad': 1.2, 'memory_windows': 2},
    '회대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 2.0, 'keep_area_local': 32.0, 'force_floor_local': 2.4, 'force_floor_grad': 1.4, 'memory_windows': 2},
    '검대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 2.6, 'keep_area_local': 38.0, 'force_floor_local': 2.5, 'force_floor_grad': 1.5, 'memory_windows': 2},
    '그마': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 2.0, 'keep_area_local': 32.0, 'force_floor_local': 2.4, 'force_floor_grad': 1.4, 'memory_windows': 2},
    '207회바': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 2.0, 'keep_area_local': 32.0, 'force_floor_local': 2.4, 'force_floor_grad': 1.4, 'memory_windows': 2},
    '흰책상': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0, 'keep_mean_local': 1.8, 'keep_area_local': 28.0, 'force_floor_local': 2.2, 'force_floor_grad': 1.3, 'memory_windows': 2},
    '나타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'clear_mean': 3.0, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0, 'keep_mean_local': 1.8, 'keep_area_local': 28.0, 'force_floor_local': 2.2, 'force_floor_grad': 1.3, 'memory_windows': 2},
    '회타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'clear_mean': 3.0, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0, 'keep_mean_local': 2.0, 'keep_area_local': 32.0, 'force_floor_local': 2.4, 'force_floor_grad': 1.4, 'memory_windows': 2},
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
REVERSE_DIRECTION_FLOORS = {'검대'}

RAW_RE = re.compile(
    r"Time\s*=?\s*(?P<time>[-+]?\d+(?:\.\d+)?)s?\s*[\|,]?\s*"
    r".*?BaseRaw\s*=?\s*(?P<base>[-+]?\d+(?:\.\d+)?)"
    r".*?CurrentRaw\s*=?\s*(?P<current>[-+]?\d+(?:\.\d+)?)"
    r".*?Change\s*=?\s*(?P<change>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE
)
GRAPH_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def ask_path(prompt, default=""):
    msg = f"{prompt}"
    if default:
        msg += f" [{default}]"
    msg += " ▶ "
    v = input(msg).strip().strip('"')
    return v if v else default

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

def apply_memory(predictions, floor_type):
    if not predictions:
        return predictions
    params = FLOOR_THRESHOLDS[floor_type]
    memory = 0
    for p in predictions:
        keep_cond = (p['mean_local_change'] >= params['keep_mean_local'] and p['area_local_change'] >= params['keep_area_local'])
        if p['label'] == '액체':
            memory = params['memory_windows']
            continue
        if memory > 0 and keep_cond:
            p['label'] = '액체'
            p['prob'] = max(p['prob'], 68.0)
            memory -= 1
        else:
            memory = max(0, memory - 1)
    return predictions

def predict(buffer, changes, baseline, floor_type, model, classes):
    predictions = []
    max_start = len(buffer) - WINDOW
    if max_start < 0:
        return []

    params = FLOOR_THRESHOLDS[floor_type]
    clean_buffer = remove_spikes(buffer)
    changes = np.abs(clean_buffer - baseline).astype(np.float32)
    sig_smooth, change_gradient, floor_ref = build_recent_floor_reference(clean_buffer, baseline, floor_type)

    floor_idx = np.where(classes == '바닥')[0][0]
    liquid_idx = np.where(classes == '액체')[0][0]

    for start_idx in range(0, max_start + 1, STEP):
        end_idx = start_idx + WINDOW
        window = create_window(clean_buffer, start_idx, baseline, floor_type)
        if window is None:
            continue

        smooth_w = sig_smooth[start_idx:end_idx]
        ref_w = floor_ref[start_idx:end_idx]
        diff_w = smooth_w - ref_w
        window_changes = changes[start_idx:end_idx]
        window_gradient = change_gradient[start_idx:end_idx]

        moving_ref = compute_moving_reference(sig_smooth, start_idx, baseline)
        local_abs = np.abs(smooth_w - moving_ref)
        mean_local_change = float(np.mean(local_abs))
        area_local_change = float(np.sum(local_abs))
        avg_change = float(np.mean(window_changes))
        max_change = float(np.max(window_changes))
        max_gradient = float(np.max(np.abs(window_gradient)))
        mean_floor_diff = float(np.mean(np.abs(diff_w)))

        pred = model.predict(np.transpose(window, (0, 2, 1)), verbose=0)[0]
        floor_prob = float(pred[floor_idx] * 100.0)
        liquid_prob = float(pred[liquid_idx] * 100.0)

        force_floor = mean_local_change <= params['force_floor_local'] and max_gradient <= params['force_floor_grad']
        strong_liquid = max_change >= params['change_strong'] and mean_local_change >= params['keep_mean_local']
        weak_floor = avg_change < params['change_low'] and mean_floor_diff < params['clear_mean']

        if force_floor:
            final_label = '바닥'
            final_prob = max(floor_prob, 88.0)
        elif strong_liquid:
            final_label = '액체'
            final_prob = max(liquid_prob, 82.0)
        elif liquid_prob >= 75.0 and mean_local_change >= params['keep_mean_local']:
            final_label = '액체'
            final_prob = liquid_prob
        elif weak_floor:
            final_label = '바닥'
            final_prob = max(floor_prob, 85.0)
        else:
            final_label = '액체' if liquid_prob >= floor_prob else '바닥'
            final_prob = max(liquid_prob, floor_prob)

        predictions.append({
            'start': start_idx, 'end': end_idx, 'label': final_label, 'prob': final_prob,
            'floor_prob': floor_prob, 'liquid_prob': liquid_prob,
            'max_change': max_change, 'avg_change': avg_change,
            'mean_local_change': mean_local_change, 'area_local_change': area_local_change,
            'max_gradient': max_gradient, 'mean_floor_diff': mean_floor_diff
        })

    return apply_memory(predictions, floor_type)

def summarize_predictions(predictions):
    cnt = Counter([p['label'] for p in predictions])
    print(f"\n최종 요약: 바닥 {cnt.get('바닥', 0)} windows / 액체 {cnt.get('액체', 0)} windows")

def save_outputs(predictions, buffer, changes, baseline, floor_type, src_csv_path, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    src_name = os.path.splitext(os.path.basename(src_csv_path))[0]
    out_csv = os.path.join(result_dir, f"{src_name}_{floor_type}_binary_predictions.csv")
    out_png = os.path.join(result_dir, f"{src_name}_{floor_type}_binary_plot.png")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["window_index", "start", "end", "label", "prob", "floor_prob", "liquid_prob", "max_change", "avg_change", "mean_local_change", "area_local_change", "max_gradient", "mean_floor_diff"])
        for i, p in enumerate(predictions):
            writer.writerow([i, p["start"], p["end"], p["label"], round(float(p["prob"]), 4), round(float(p["floor_prob"]), 4), round(float(p["liquid_prob"]), 4), round(float(p["max_change"]), 4), round(float(p["avg_change"]), 4), round(float(p["mean_local_change"]), 4), round(float(p["area_local_change"]), 4), round(float(p["max_gradient"]), 4), round(float(p["mean_floor_diff"]), 4)])

    FLOOR_COLOR = '#E8E8E8'
    LIQUID_COLOR = '#5C2F0F'
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.6, 1, 1])

    ax1 = fig.add_subplot(gs[0]); ax1_twin = ax1.twinx()
    ax1.plot(buffer, 'b-', lw=2, label='CurrentRaw')
    ax1.axhline(baseline, color='green', ls='--', lw=2, label=f'Base: {baseline:.1f}')
    ax1_twin.plot(changes, 'r:', alpha=0.7, lw=1.8, label='Global Change')
    for r in predictions:
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        alpha = 0.25 if r['label'] == '바닥' else 0.5
        ax1.axvspan(r['start'], min(r['end'], len(buffer)), color=color, alpha=alpha)
    ax1.set_title(f'{floor_type} 바닥 - LSTM 실시간 액체 감지 결과')
    ax1.legend(loc='upper left'); ax1_twin.legend(loc='upper right'); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    bar_colors = [FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR for r in predictions]
    ax2.bar(range(len(predictions)), [r['prob'] for r in predictions], color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_ylim(0, 105); ax2.set_title('신뢰도'); ax2.grid(alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[2])
    ax3.plot([r['max_gradient'] for r in predictions], 'purple', alpha=0.8, label='Gradient')
    ax3.plot([r['mean_local_change'] for r in predictions], color='orange', alpha=0.8, label='Moving Baseline Change')
    ax3.axhline(FLOOR_THRESHOLDS[floor_type]['gradient'], color='red', ls='--', lw=2, label='Gradient 기준')
    ax3.legend(loc='upper right'); ax3.set_title('Gradient / Moving Baseline Change'); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[3])
    for i, r in enumerate(predictions):
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        ax4.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', lw=1.2))
        text_color = 'black' if r['label'] == '바닥' else 'white'
        ax4.text(i + 0.5, 0.5, '바' if r['label'] == '바닥' else '액', ha='center', va='center', fontsize=9, color=text_color, fontweight='bold')
    ax4.set_xlim(0, len(predictions)); ax4.set_ylim(0, 1); ax4.set_yticks([]); ax4.set_title('예측 시퀀스')
    legend_handles = [Patch(facecolor=FLOOR_COLOR, edgecolor='black', label='바닥'), Patch(facecolor=LIQUID_COLOR, edgecolor='black', label='액체')]
    ax4.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("예측 CSV 저장:", out_csv)
    print("예측 그림 저장:", out_png)

if __name__ == '__main__':
    default_root = r"C:\Users\MASL\Desktop\3차리뉴얼db"
    root_dir = ask_path("루트 폴더", default_root)
    save_dir = ask_path("모델 폴더", os.path.join(root_dir, "kumoh_binary_model_save"))
    scenario_dir = ask_path("시나리오 폴더", os.path.join(root_dir, "scenario_predict"))
    result_dir = ask_path("결과 저장 폴더", os.path.join(root_dir, "predict_results_binary"))

    encoder_path = os.path.join(save_dir, 'encoder.pkl')
    encoder = pickle.load(open(encoder_path, 'rb'))
    classes = encoder.classes_

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

        model_path = os.path.join(save_dir, f"model_{floor_type}.keras")
        if not os.path.exists(model_path):
            print('모델 파일 없음:', model_path)
            continue
        model = tf.keras.models.load_model(model_path)

        scenario_files = sorted([f for f in os.listdir(scenario_dir) if f.lower().endswith('.csv')]) if os.path.isdir(scenario_dir) else []
        if scenario_files:
            for i, name in enumerate(scenario_files, start=1):
                print(f'[{i}] {name}')
        sel = input('예측할 CSV 번호 또는 전체 경로 ▶ ').strip().strip('"')
        if sel.isdigit() and scenario_files and (1 <= int(sel) <= len(scenario_files)):
            csv_path = os.path.join(scenario_dir, scenario_files[int(sel)-1])
        else:
            csv_path = sel
        if not os.path.exists(csv_path):
            print('파일을 찾을 수 없습니다.')
            continue

        baseline, buffer, changes = read_csv_with_fixed_baseline(csv_path)
        if baseline is None:
            print('CSV를 읽지 못했습니다.')
            continue

        preds = predict(buffer, changes, baseline, floor_type, model, classes)
        for i, p in enumerate(preds[:25], start=1):
            print(f"[{i:02d}] {p['start']:4d}-{p['end']:4d} | {p['label']:>4s} | prob={p['prob']:5.1f} | floor={p['floor_prob']:5.1f} | liquid={p['liquid_prob']:5.1f} | maxchg={p['max_change']:6.2f} | mbase={p['mean_local_change']:5.2f} | grad={p['max_gradient']:5.2f}")
        summarize_predictions(preds)
        save_outputs(preds, buffer, changes, baseline, floor_type, csv_path, result_dir)
