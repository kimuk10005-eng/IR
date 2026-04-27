# -*- coding: utf-8 -*-
"""
3차 상태유지 실험용 예측 코드 v2 (이진 분류: 바닥/액체)

핵심
- 모델 출력은 '바닥', '액체' 두 개만 사용
- 상태는 FLOOR / LIQUID_HOLD 만 유지
- 첫 진입은 보수적으로: 충분한 floor history + 연속 이벤트 근거 필요
- 액체 plateau는 anchor baseline 대비 offset이 충분히 클 때만 유지
- 해제는 연속 release 조건이 쌓일 때만 FLOOR 복귀
- 결과 png와 txt 로그를 같은 폴더에 저장
"""

import os
import csv
import re
import pickle
from collections import Counter, deque
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
    '황대': {'grad_enter': 0.65, 'change_enter': 5.5, 'hold_offset': 7.0, 'hold_release': 4.2, 'flat_std': 1.00, 'release_need': 3, 'enter_need': 2},
    '회대': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 8.5, 'hold_release': 5.0, 'flat_std': 1.10, 'release_need': 3, 'enter_need': 2},
    '나타': {'grad_enter': 0.85, 'change_enter': 6.5, 'hold_offset': 9.0, 'hold_release': 5.5, 'flat_std': 1.20, 'release_need': 3, 'enter_need': 2},
    '회타': {'grad_enter': 0.85, 'change_enter': 6.5, 'hold_offset': 9.0, 'hold_release': 5.5, 'flat_std': 1.20, 'release_need': 3, 'enter_need': 2},
    '검대': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 8.5, 'hold_release': 5.0, 'flat_std': 1.10, 'release_need': 3, 'enter_need': 2},
    '그마': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 8.5, 'hold_release': 5.0, 'flat_std': 1.10, 'release_need': 3, 'enter_need': 2},
    '207회바': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 8.5, 'hold_release': 5.0, 'flat_std': 1.10, 'release_need': 3, 'enter_need': 2},
    '흰책상': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 8.5, 'hold_release': 5.0, 'flat_std': 1.10, 'release_need': 3, 'enter_need': 2},
    '나무': {'grad_enter': 0.75, 'change_enter': 6.0, 'hold_offset': 8.5, 'hold_release': 5.0, 'flat_std': 1.10, 'release_need': 3, 'enter_need': 2},
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

FLOOR_COLOR = '#E8E8E8'
LIQUID_COLOR = '#7A4B1F'


def to_float(x):
    try:
        x = str(x).strip()
        return float(x) if x != '' else None
    except Exception:
        return None


def log(msg, fp=None):
    print(msg)
    if fp is not None:
        fp.write(msg + '\n')
        fp.flush()


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
        if abs(float(sig[i]) - local_med) > k * local_std and abs(float(sig[i]) - float(sig[i-1])) > 1.2 * local_std and abs(float(sig[i]) - float(sig[i+1])) > 1.2 * local_std:
            sig[i] = np.float32(0.5 * (sig[i - 1] + sig[i + 1]))
    return sig


def read_csv_with_fixed_baseline(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    def parse_new_format(lines):
        baseline_values = []
        row_bases = []
        data_points = []
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

    def parse_old_format(lines):
        baseline_values = []
        row_bases = []
        data_points = []
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
            result = parse_new_format(lines)
            if result[0] is not None:
                return result
            result = parse_old_format(lines)
            if result[0] is not None:
                return result
        except Exception:
            continue
    return None, None, None


def create_window(full_signal, start_idx, baseline, floor_type):
    if start_idx + WINDOW > len(full_signal):
        return None
    sig = full_signal[start_idx:start_idx + WINDOW]
    norm_sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    deriv1 = np.diff(norm_sig, prepend=norm_sig[0])
    deriv1 = np.convolve(deriv1, np.ones(3) / 3, 'same')
    deriv2 = np.diff(deriv1, prepend=deriv1[0])
    base_eff = baseline if len(full_signal) < 20 else 0.7 * baseline + 0.3 * float(np.mean(full_signal[:20]))
    context_start = max(0, start_idx - 5)
    context_end = min(len(full_signal), start_idx + WINDOW + 5)
    recent_mean = float(np.mean(full_signal[context_start:context_end]))
    change_raw = base_eff - recent_mean if floor_type in REVERSE_DIRECTION_FLOORS else recent_mean - base_eff
    direction = np.clip(change_raw / 300.0, -1.0, 1.0)
    direction_channel = np.full((WINDOW,), direction, dtype=np.float32)
    window = np.stack([norm_sig, deriv1, deriv2, direction_channel]).astype(np.float32)
    return window[np.newaxis, :, :]


def compute_window_features(sig_smooth, start_idx, baseline):
    sig_w = sig_smooth[start_idx:start_idx + WINDOW]
    grad_w = np.gradient(sig_w)
    abs_grad = np.abs(grad_w)
    avg_change = float(np.mean(np.abs(sig_w - baseline)))
    max_change = float(np.max(np.abs(sig_w - baseline)))
    mean_raw = float(np.mean(sig_w))
    max_grad = float(np.max(abs_grad))
    edge_ratio = float(np.mean(abs_grad >= 1.0))
    flat_std = float(np.std(sig_w))
    return {
        'sig_w': sig_w,
        'avg_change': avg_change,
        'max_change': max_change,
        'mean_raw': mean_raw,
        'max_grad': max_grad,
        'edge_ratio': edge_ratio,
        'flat_std': flat_std,
    }


def stable_floor_cond(feat, params):
    return (
        feat['avg_change'] <= params['hold_release'] and
        feat['max_grad'] <= params['grad_enter'] and
        feat['flat_std'] <= params['flat_std'] * 2.0
    )


def enter_event_cond(feat, liquid_prob, params):
    strong_edge = feat['max_grad'] >= params['grad_enter'] * 1.25 and feat['edge_ratio'] >= 0.25
    strong_change = feat['avg_change'] >= params['change_enter']
    model_help = liquid_prob >= 0.58
    very_strong = feat['max_grad'] >= params['grad_enter'] * 2.4 and feat['avg_change'] >= params['change_enter'] * 1.25 and feat['edge_ratio'] >= 0.22
    enter = (strong_edge and strong_change) or (strong_edge and model_help) or very_strong
    return enter, very_strong


def smooth_labels(predictions):
    if not predictions:
        return predictions
    labels = [p['final'] for p in predictions]
    # remove 1-window liquid spikes
    for i in range(1, len(labels) - 1):
        if labels[i] == '액체' and labels[i - 1] == '바닥' and labels[i + 1] == '바닥':
            labels[i] = '바닥'
    # fill 1-window floor hole inside liquid region
    for i in range(1, len(labels) - 1):
        if labels[i] == '바닥' and labels[i - 1] == '액체' and labels[i + 1] == '액체':
            labels[i] = '액체'
    for p, lab in zip(predictions, labels):
        p['final'] = lab
    return predictions


def predict(buffer, baseline, floor_type, model, encoder):
    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    sig_smooth = smooth_signal(remove_spikes(buffer))

    classes = list(encoder.classes_)
    floor_idx = classes.index('바닥')
    liquid_idx = classes.index('액체')

    predictions = []
    floor_history = deque(maxlen=4)
    state = 'FLOOR'
    anchor = baseline
    enter_count = 0
    release_count = 0

    max_start = len(sig_smooth) - WINDOW
    for start_idx in range(0, max_start + 1, STEP):
        feat = compute_window_features(sig_smooth, start_idx, baseline)
        window = create_window(sig_smooth, start_idx, baseline, floor_type)
        pred = model.predict(np.transpose(window, (0, 2, 1)), verbose=0)[0]
        liquid_prob = float(pred[liquid_idx] * 100.0)
        floor_prob = float(pred[floor_idx] * 100.0)

        if stable_floor_cond(feat, params):
            floor_history.append(feat['mean_raw'])

        enter_event, very_strong = enter_event_cond(feat, liquid_prob / 100.0, params)
        if state == 'FLOOR':
            if len(floor_history) >= 2 and enter_event:
                enter_count += 1
            else:
                enter_count = 0

            if len(floor_history) >= 2 and (enter_count >= params['enter_need'] or (very_strong and enter_count >= 1)):
                anchor = float(np.mean(list(floor_history)[-2:]))
                state = 'LIQUID_HOLD'
                release_count = 0
                final = '액체'
            else:
                final = '바닥'
        else:
            mean_anchor_offset = float(np.mean(np.abs(feat['sig_w'] - anchor)))
            max_anchor_offset = float(np.max(np.abs(feat['sig_w'] - anchor)))
            keep_by_offset = mean_anchor_offset >= params['hold_offset']
            keep_by_event = enter_event and feat['avg_change'] >= params['change_enter'] * 0.9
            keep_by_model = liquid_prob >= 62.0 and mean_anchor_offset >= params['hold_release'] * 1.1
            
            release_ready = (
                mean_anchor_offset <= params['hold_release'] and
                feat['max_grad'] <= params['grad_enter'] * 1.05 and
                stable_floor_cond(feat, params)
            )

            if keep_by_offset or keep_by_event or keep_by_model:
                final = '액체'
                release_count = 0
            else:
                if release_ready:
                    release_count += 1
                else:
                    release_count = 0
                if release_count >= params['release_need']:
                    state = 'FLOOR'
                    enter_count = 0
                    floor_history.append(feat['mean_raw'])
                    final = '바닥'
                else:
                    final = '액체'
        
        mean_anchor_offset = float(np.mean(np.abs(feat['sig_w'] - anchor)))
        max_anchor_offset = float(np.max(np.abs(feat['sig_w'] - anchor)))
        predictions.append({
            'start': start_idx,
            'end': start_idx + WINDOW,
            'state': state,
            'final': final,
            'floor_prob': floor_prob,
            'liquid_prob': liquid_prob,
            'avg_change': feat['avg_change'],
            'max_change': feat['max_change'],
            'max_grad': feat['max_grad'],
            'edge_ratio': feat['edge_ratio'],
            'flat_std': feat['flat_std'],
            'anchor': float(anchor),
            'mean_anchor_offset': mean_anchor_offset,
            'max_anchor_offset': max_anchor_offset,
        })

    return smooth_labels(predictions)


def save_results(predictions, buffer, changes, baseline, floor_type, csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    log_path = os.path.join(out_dir, f'{stem}_log.txt')
    img_path = os.path.join(out_dir, f'{stem}_plot.png')

    with open(log_path, 'w', encoding='utf-8') as fp:
        log(f'[CSV] {csv_path}', fp)
        log(f'[결과 폴더] {os.path.abspath(out_dir)}', fp)
        log(f'[바닥] {floor_type}', fp)
        log(f'[baseline] {baseline:.3f}', fp)
        total = len(predictions)
        floor_n = sum(1 for p in predictions if p['final'] == '바닥')
        liquid_n = total - floor_n
        log(f'전체 window 수: {total}', fp)
        log(f'바닥: {floor_n}개 ({floor_n/total*100:.1f}%)', fp)
        log(f'액체: {liquid_n}개 ({liquid_n/total*100:.1f}%)', fp)
        log('\n[상세 window]', fp)
        for i, p in enumerate(predictions, start=1):
            log(
                f"[{i:03d}] {p['start']:4d}-{p['end']:4d} | final={p['final']:>4s} | state={p['state']:>11s} | "
                f"pL={p['liquid_prob']:5.1f} | pF={p['floor_prob']:5.1f} | "
                f"anch_ofs={p['mean_anchor_offset']:6.2f}/{p['max_anchor_offset']:6.2f} | "
                f"chg={p['avg_change']:6.2f} | grad={p['max_grad']:5.2f} | edge={p['edge_ratio']:.2f} | anchor={p['anchor']:.2f}",
                fp,
            )

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.6, 1.2, 1.2, 1.4], hspace=0.25)
    fig.suptitle(f'{floor_type} 바닥 - 상태유지 기반 액체 감지 결과 (이진)', fontsize=22, fontweight='bold', y=0.97)

    ax1 = fig.add_subplot(gs[0])
    for p in predictions:
        color = FLOOR_COLOR if p['final'] == '바닥' else LIQUID_COLOR
        ax1.axvspan(p['start'], p['end'], color=color, alpha=0.25)
    ax1.plot(buffer, color='blue', lw=2, label='CurrentRaw')
    ax1.axhline(baseline, color='green', lw=2, ls='--', label=f'Base: {baseline:.1f}')
    ax1.grid(alpha=0.25)
    ax1.legend(loc='upper left')
    ax1r = ax1.twinx()
    ax1r.plot(changes, color='red', lw=2, ls=':', alpha=0.75, label='Global Change')
    ax1r.legend(loc='upper right')

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(range(len(predictions)), [p['liquid_prob'] for p in predictions], color=[FLOOR_COLOR if p['final']=='바닥' else LIQUID_COLOR for p in predictions], edgecolor='black', linewidth=0.4)
    ax2.set_ylim(0, 100)
    ax2.set_title('액체 확률')
    ax2.grid(axis='y', alpha=0.25)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot([p['max_grad'] for p in predictions], color='purple', label='Gradient (최대 변화율)')
    ax3.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['grad_enter'], color='red', ls='--', label='Gradient 기준')
    ax3r = ax3.twinx()
    ax3r.plot([p['mean_anchor_offset'] for p in predictions], color='orange', label='Anchor Offset')
    ax3r.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['hold_offset'], color='brown', ls=':', label='Hold 기준')
    l1, t1 = ax3.get_legend_handles_labels()
    l2, t2 = ax3r.get_legend_handles_labels()
    ax3.legend(l1+l2, t1+t2, loc='upper right')
    ax3.set_title('Gradient / Anchor Offset')
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[3])
    for i, p in enumerate(predictions):
        color = FLOOR_COLOR if p['final'] == '바닥' else LIQUID_COLOR
        ax4.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='k', lw=1.0))
        ax4.text(i + 0.5, 0.5, '바' if p['final']=='바닥' else '액', ha='center', va='center', fontsize=8, fontweight='bold', color='black' if p['final']=='바닥' else 'white')
    ax4.set_xlim(0, len(predictions))
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_title('예측 시퀀스')
    ax4.legend(handles=[Patch(facecolor=FLOOR_COLOR, edgecolor='black', label='바닥'), Patch(facecolor=LIQUID_COLOR, edgecolor='black', label='액체')], loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return log_path, img_path


def main():
    print('\n' + '=' * 80)
    print('LSTM 액체 감지 예측기 - 상태유지 v2 (이진)'.center(80))
    print('=' * 80 + '\n')
    print(f'[현재 작업 폴더] {os.getcwd()}')
    print(f'[모델 폴더] {os.path.abspath(SAVE_DIR)}')
    print(f'[결과 루트] {os.path.abspath(RESULT_ROOT)}')

    if not os.path.exists(ENCODER_PATH):
        print(f'encoder.pkl을 찾지 못했습니다: {os.path.abspath(ENCODER_PATH)}')
        return
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    print(f'[encoder classes] {list(encoder.classes_)}')

    models = {}
    for floor, path in FLOOR_MODELS.items():
        if os.path.exists(path):
            try:
                models[floor] = tf.keras.models.load_model(path)
                print(f'모델 로드 완료: {floor}')
            except Exception as e:
                print(f'{floor} 모델 로드 실패: {e}')

    while True:
        print(f"\n지원 바닥: {', '.join(models.keys()) if models else '없음'} | 종료: q")
        cmd = input('바닥 타입 ▶ ').strip()
        if cmd.lower() in ['q', 'quit', 'exit', '']:
            print('종료합니다.')
            break
        floor_type = find_floor_type(cmd)
        if floor_type not in models:
            print('해당 바닥 모델이 없습니다.')
            continue

        csv_path = input('CSV 경로 ▶ ').strip().strip('"')
        if not os.path.exists(csv_path):
            print('파일을 찾을 수 없습니다.')
            continue

        baseline, buffer, changes = read_csv_with_fixed_baseline(csv_path)
        if baseline is None:
            print('CSV 파싱 실패')
            continue

        preds = predict(buffer, baseline, floor_type, models[floor_type], encoder)
        out_dir = os.path.join(RESULT_ROOT, f'{floor_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        log_path, img_path = save_results(preds, buffer, changes, baseline, floor_type, csv_path, out_dir)
        print(f'[저장 로그] {os.path.abspath(log_path)}')
        print(f'[저장 그림] {os.path.abspath(img_path)}')
        floor_n = sum(1 for p in preds if p['final'] == '바닥')
        liquid_n = len(preds) - floor_n
        print(f'총 {len(preds)}개 | 바닥 {floor_n} | 액체 {liquid_n}')


if __name__ == '__main__':
    main()
