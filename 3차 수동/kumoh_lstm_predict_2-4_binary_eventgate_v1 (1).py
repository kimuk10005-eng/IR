# -*- coding: utf-8 -*-
"""
2-4 구조 유지 + binary(바닥/액체) + event gate 중심 예측 코드
- 상태 유지형 hold 제거
- event seed를 먼저 찾고, seed 주변만 짧게 plateau bridge 허용
- 결과 폴더에 로그 txt + plot png 저장
"""

import os, csv, re, pickle, datetime
from collections import Counter
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = './kumoh_lstm_model_save'
ENCODER_PATH = os.path.join(SAVE_DIR, 'encoder_binary.pkl')
WINDOW = 15
STEP = 5
SMOOTH_K = 5
RESULT_BASE = './predict_results'
os.makedirs(RESULT_BASE, exist_ok=True)

FLOOR_ALIASES = {
    '나무': ['나무', 'wood', '나'], '황대': ['황대', '황색대리석', '황색', 'yellow', '황'],
    '회대': ['회대', '회색대리석', '회색', 'gray', 'grey', '회'], '검대': ['검대', '검정색대리석', '검정', 'black', '검'],
    '그마': ['그마', 'greymarble'], '207회바': ['회바', '207', '207greyfloor', 'greyfloor'],
    '흰책상': ['흰책상', 'white', 'whitedesk'], '나타': ['나타'], '회타': ['회타']
}
REVERSE_DIRECTION_FLOORS = {'검대'}
RAW_RE = re.compile(r"Time=(?P<time>[-+]?\d+(?:\.\d+)?)s\s*\|\s*BaseRaw=(?P<base>[-+]?\d+(?:\.\d+)?)\s*\|\s*CurrentRaw=(?P<current>[-+]?\d+(?:\.\d+)?)\s*\|\s*Change=(?P<change>[-+]?\d+(?:\.\d+)?)")

FLOOR_THRESHOLDS = {
    '나무': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.030, 'ref_grad': 0.80, 'ref_band': 6.5},
    '황대': {'gradient': 0.6, 'change_low': 5.0, 'change_strong': 15.0, 'plateau_mean': 7.0, 'plateau_med': 6.0, 'clear_mean': 2.3, 'alpha': 0.028, 'ref_grad': 0.75, 'ref_band': 6.0},
    '회대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0},
    '검대': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0},
    '그마': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0},
    '207회바': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0},
    '흰책상': {'gradient': 0.7, 'change_low': 6.0, 'change_strong': 18.0, 'plateau_mean': 9.0, 'plateau_med': 7.0, 'clear_mean': 2.8, 'alpha': 0.026, 'ref_grad': 0.75, 'ref_band': 6.0},
    '나타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'plateau_mean': 10.0, 'plateau_med': 8.0, 'clear_mean': 3.0, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0},
    '회타': {'gradient': 0.8, 'change_low': 6.5, 'change_strong': 19.0, 'plateau_mean': 10.0, 'plateau_med': 8.0, 'clear_mean': 3.0, 'alpha': 0.020, 'ref_grad': 0.65, 'ref_band': 5.0},
}

try:
    encoder = pickle.load(open(ENCODER_PATH, 'rb'))
    classes = encoder.classes_
except Exception as e:
    raise SystemExit(f'encoder_binary.pkl 로드 실패: {e}')


def model_path_for(floor):
    return os.path.join(SAVE_DIR, f'model_{floor}_binary.keras')


def to_float(x):
    try:
        return float(str(x).strip())
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


def read_csv_with_fixed_baseline(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
    def parse_new(lines):
        baseline_values, row_bases, data_points = [], [], []
        reader = csv.DictReader(lines)
        fields = [str(name).strip() for name in (reader.fieldnames or [])]
        if 'record_type' not in fields:
            return None, None, None
        for row in reader:
            rt = str(row.get('record_type', '')).strip().lower()
            baseline_raw = to_float(row.get('baseline_raw', ''))
            base_raw = to_float(row.get('base_raw', ''))
            current_raw = to_float(row.get('current_raw', ''))
            raw_data = str(row.get('raw_data', '')).strip()
            if rt == 'baseline':
                if baseline_raw is not None:
                    baseline_values.append(baseline_raw)
                elif base_raw is not None:
                    baseline_values.append(base_raw)
                continue
            if rt == 'data':
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
        base = float(np.mean(baseline_values)) if baseline_values else (float(np.mean(row_bases)) if row_bases else None)
        if base is None:
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
                base_raw = None; current_raw = None
                for i in range(len(parts)-1):
                    if parts[i] == 'BaseRaw':
                        base_raw = to_float(parts[i+1])
                    elif parts[i] == 'CurrentRaw':
                        current_raw = to_float(parts[i+1])
                if current_raw is not None:
                    data_points.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)
        if len(data_points) < WINDOW:
            return None, None, None
        base = final_base if final_base is not None else (float(np.mean(baseline_values)) if baseline_values else (float(np.mean(row_bases)) if row_bases else None))
        if base is None:
            return None, None, None
        buffer = np.array(data_points, dtype=np.float32)
        changes = np.abs(buffer - base).astype(np.float32)
        return base, buffer, changes
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


def create_window(full_signal, start_idx, baseline, floor_type):
    sig = full_signal[start_idx:start_idx+WINDOW]
    if len(sig) != WINDOW:
        return None
    norm_sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    deriv1 = np.diff(norm_sig, prepend=norm_sig[0])
    deriv1 = np.convolve(deriv1, np.ones(3)/3, 'same')
    deriv2 = np.diff(deriv1, prepend=deriv1[0])
    base_eff = baseline if len(full_signal) < 20 else 0.7*baseline + 0.3*np.mean(full_signal[:20])
    context_start = max(0, start_idx-5)
    context_end = min(len(full_signal), start_idx+WINDOW+5)
    recent_mean = np.mean(full_signal[context_start:context_end])
    change_raw = base_eff - recent_mean if floor_type in REVERSE_DIRECTION_FLOORS else recent_mean - base_eff
    direction = np.clip(change_raw/300.0, -1.0, 1.0)
    direction_channel = np.full((WINDOW,), direction)
    return np.stack([norm_sig, deriv1, deriv2, direction_channel])[np.newaxis, :, :]


def run_event_gate(predictions, params):
    n = len(predictions)
    seed = np.zeros(n, dtype=bool)
    for i, p in enumerate(predictions):
        seed[i] = (
            (p['max_gradient'] >= params['gradient'] * 1.8 and p['max_change'] >= params['change_low']) or
            (p['edge_ratio'] >= 0.40 and p['max_change'] >= params['change_low']) or
            (p['liquid_prob'] >= 85.0 and p['max_change'] >= params['change_strong'])
        )
    labels = ['바닥'] * n
    # short bridge around seeds only
    for i in range(n):
        if not seed[i]:
            continue
        left = max(0, i-1); right = min(n, i+2)
        for k in range(left, right):
            p = predictions[k]
            if p['max_change'] >= params['change_low'] and (p['mean_floor_diff'] >= params['plateau_mean'] or p['max_gradient'] >= params['gradient']):
                labels[k] = '액체'
    # remove isolated single windows
    i = 0
    while i < n:
        if labels[i] == '바닥':
            i += 1; continue
        j = i
        while j+1 < n and labels[j+1] == '액체':
            j += 1
        run_len = j-i+1
        run_has_seed = np.any(seed[i:j+1])
        if run_len == 1 and not run_has_seed:
            labels[i] = '바닥'
        if run_len > 4 and not np.any(seed[max(0, i-1):min(n, j+2)]):
            for k in range(i, j+1):
                labels[k] = '바닥'
        i = j+1
    return labels, seed


def predict(buffer, changes, baseline, floor_type, model):
    predictions = []
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    max_start = len(buffer) - WINDOW
    if max_start < 0:
        return []
    clean_buffer = remove_spikes(buffer)
    changes = np.abs(clean_buffer - baseline).astype(np.float32)
    sig_smooth, change_gradient, floor_ref = build_recent_floor_reference(clean_buffer, baseline, floor_type)
    floor_idx = int(np.where(classes == '바닥')[0][0]) if '바닥' in classes else 0
    liquid_idx = int(np.where(classes == '액체')[0][0]) if '액체' in classes else 1
    for start_idx in range(0, max_start + 1, STEP):
        window = create_window(clean_buffer, start_idx, baseline, floor_type)
        if window is None:
            continue
        smooth_w = sig_smooth[start_idx:start_idx+WINDOW]
        ref_w = floor_ref[start_idx:start_idx+WINDOW]
        diff_w = smooth_w - ref_w
        window_changes = changes[start_idx:start_idx+WINDOW]
        window_gradient = change_gradient[start_idx:start_idx+WINDOW]
        max_change = float(np.max(window_changes))
        avg_change = float(np.mean(window_changes))
        max_gradient = float(np.max(np.abs(window_gradient)))
        edge_ratio = float(np.mean(np.abs(window_gradient) >= params['gradient']))
        mean_floor_diff = float(np.mean(np.abs(diff_w)))
        median_floor_diff = float(np.abs(np.median(diff_w)))
        if model:
            pred = model.predict(np.transpose(window, (0,2,1)), verbose=0)[0]
        else:
            pred = np.zeros(len(classes), dtype=np.float32)
            pred[floor_idx] = 1.0
        floor_prob = float(pred[floor_idx] * 100.0)
        liquid_prob = float(pred[liquid_idx] * 100.0)
        predictions.append({
            'start': start_idx, 'end': start_idx + WINDOW,
            'max_change': max_change, 'avg_change': avg_change,
            'max_gradient': max_gradient, 'edge_ratio': edge_ratio,
            'mean_floor_diff': mean_floor_diff, 'median_floor_diff': median_floor_diff,
            'floor_prob': floor_prob, 'liquid_prob': liquid_prob,
        })
    labels, seeds = run_event_gate(predictions, params)
    for p, lab, sd in zip(predictions, labels, seeds):
        p['label'] = lab
        p['seed'] = bool(sd)
        p['prob'] = p['liquid_prob'] if lab == '액체' else p['floor_prob']
    return predictions


def save_and_visualize(predictions, buffer, changes, baseline, floor_type, csv_path):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.abspath(os.path.join(RESULT_BASE, f'{floor_type}_{ts}'))
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f'{stem}_plot.png')
    log_path = os.path.join(out_dir, f'{stem}_log.txt')

    FLOOR_COLOR = '#e0e0e0'; LIQUID_COLOR = '#7b4a1e'; SEED_COLOR = '#c0392b'
    fig = plt.figure(figsize=(19,9))
    gs = fig.add_gridspec(4,1,height_ratios=[3.0,2.2,1.4,1.8], hspace=0.30)
    fig.suptitle(f'{floor_type} 바닥 - LSTM 실시간 액체 감지 결과', fontsize=20, fontweight='bold', y=0.97)
    ax1 = fig.add_subplot(gs[0])
    for r in predictions:
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        ax1.axvspan(r['start'], r['end'], color=color, alpha=0.22)
        if r['seed']:
            ax1.axvspan(r['start'], r['end'], color=SEED_COLOR, alpha=0.10)
    ax1.plot(buffer, color='blue', lw=2, label='CurrentRaw')
    ax1.axhline(baseline, color='green', lw=2, ls='--', label=f'Base: {baseline:.1f}')
    ax1.grid(alpha=0.25); ax1.legend(loc='upper left', fontsize=11)
    ax1r = ax1.twinx(); ax1r.plot(changes, color='red', lw=2, ls=':', alpha=0.75, label='Global Change'); ax1r.legend(loc='upper right', fontsize=11)
    ax2 = fig.add_subplot(gs[1])
    probs = [r['liquid_prob'] for r in predictions]
    colors = [FLOOR_COLOR if r['label']=='바닥' else LIQUID_COLOR for r in predictions]
    ax2.bar(range(len(predictions)), probs, color=colors, edgecolor='black', linewidth=0.4)
    ax2.set_ylim(0,105); ax2.set_title('액체 확률'); ax2.grid(axis='y', alpha=0.25)
    ax3 = fig.add_subplot(gs[2])
    grads = [r['max_gradient'] for r in predictions]; diffs = [r['mean_floor_diff'] for r in predictions]
    ax3.plot(grads, color='purple', alpha=0.85, label='Gradient (최대 변화율)')
    ax3.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['gradient'], color='red', ls='--', lw=2, label='Gradient 기준')
    ax3r = ax3.twinx(); ax3r.plot(diffs, color='orange', alpha=0.75, label='Mean Floor Diff'); ax3r.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['plateau_mean'], color='brown', ls=':', lw=2, label='Plateau 기준')
    l1, lb1 = ax3.get_legend_handles_labels(); l2, lb2 = ax3r.get_legend_handles_labels(); ax3.legend(l1+l2, lb1+lb2, loc='upper right', fontsize=10)
    ax3.set_title('Gradient / Floor Diff'); ax3.grid(alpha=0.25)
    ax4 = fig.add_subplot(gs[3])
    for i, r in enumerate(predictions):
        color = FLOOR_COLOR if r['label']=='바닥' else LIQUID_COLOR
        ax4.add_patch(Rectangle((i,0),1,1,facecolor=color,edgecolor='k',lw=1.2))
        txt = '바' if r['label']=='바닥' else '액'
        txt_color = 'black' if r['label']=='바닥' else 'white'
        ax4.text(i+0.5,0.5,txt,ha='center',va='center',fontweight='bold',fontsize=8,color=txt_color)
    ax4.set_xlim(0,len(predictions)); ax4.set_ylim(0,1); ax4.set_yticks([]); ax4.set_title('예측 시퀀스')
    legend_handles = [Patch(facecolor=FLOOR_COLOR, edgecolor='black', label='바닥'), Patch(facecolor=LIQUID_COLOR, edgecolor='black', label='액체')]
    ax4.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=2, fontsize=12)
    plt.tight_layout(); plt.savefig(plot_path, dpi=160, bbox_inches='tight'); plt.close(fig)

    cnt = Counter([p['label'] for p in predictions]); liquid_count = cnt.get('액체', 0)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'[CSV] {csv_path}\n[결과 폴더] {out_dir}\n[바닥] {floor_type}\n[baseline] {baseline:.3f}\n')
        f.write(f'전체 window 수: {len(predictions)}\n바닥: {cnt.get("바닥",0)}개 ({cnt.get("바닥",0)/len(predictions)*100:.1f}%)\n액체: {liquid_count}개 ({liquid_count/len(predictions)*100:.1f}%)\n\n[상세 window]\n')
        for i, p in enumerate(predictions, start=1):
            f.write(f'[{i:03d}] {p["start"]:4d}-{p["end"]:4d} | final={p["label"]:>4s} | seed={int(p["seed"])} | pL={p["liquid_prob"]:5.1f} | pF={p["floor_prob"]:5.1f} | chg={p["max_change"]:6.2f} | grad={p["max_gradient"]:5.2f} | edge={p["edge_ratio"]:.2f} | fdiff={p["mean_floor_diff"]:.2f}\n')
    return out_dir, log_path, plot_path


def main():
    print('\n' + '='*80)
    print('2-4 binary event-gate 예측기'.center(80))
    print('='*80 + '\n')
    while True:
        cmd = input('바닥 타입 ▶ ').strip()
        if cmd.lower() in ['q', 'quit', 'exit', 'ㅂㅂ', '']:
            print('종료'); break
        floor_type = find_floor_type(cmd)
        if not floor_type:
            print('지원되지 않는 바닥 타입'); continue
        model_path = model_path_for(floor_type)
        if not os.path.exists(model_path):
            print(f'모델 파일 없음: {os.path.abspath(model_path)}'); continue
        model = tf.keras.models.load_model(model_path)
        csv_path = input('CSV 경로 ▶ ').strip().strip('"')
        if not os.path.exists(csv_path):
            print('파일 없음'); continue
        baseline, buffer, changes = read_csv_with_fixed_baseline(csv_path)
        if baseline is None:
            print('CSV 파싱 실패'); continue
        preds = predict(buffer, changes, baseline, floor_type, model)
        out_dir, log_path, plot_path = save_and_visualize(preds, buffer, changes, baseline, floor_type, csv_path)
        cnt = Counter([p['label'] for p in preds])
        print(f'[결과 폴더] {out_dir}')
        print(f'총 윈도우: {len(preds)} | 바닥: {cnt.get("바닥",0)} | 액체: {cnt.get("액체",0)}')
        print(f'[로그] {log_path}')
        print(f'[플롯] {plot_path}')

if __name__ == '__main__':
    main()
