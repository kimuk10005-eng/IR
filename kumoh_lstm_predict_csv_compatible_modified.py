import numpy as np
import pickle
import tensorflow as tf
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from collections import Counter

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = './kumoh_lstm_model_save'
ENCODER_PATH = os.path.join(SAVE_DIR, 'encoder.pkl')
WINDOW = 15
STEP = 3

# False: CSV의 change 열 사용
# True : abs(current_raw - baseline)로 다시 계산
RECALC_CHANGE_FROM_BASELINE = True

FLOOR_THRESHOLDS = {
    '나무': {'gradient': 0.5, 'change': 5},
    '황대': {'gradient': 0.5, 'change': 3},
    '회대': {'gradient': 0.5, 'change': 3},
    '검대': {'gradient': 0.5, 'change': 3},
    '그마': {'gradient': 0.5, 'change': 3},
    '207회바': {'gradient': 0.5, 'change': 3},
    '흰책상': {'gradient': 0.5, 'change': 3},
    '나타': {'gradient': 0.5, 'change': 3},
    '회타': {'gradient': 0.5, 'change': 3},
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

try:
    encoder = pickle.load(open(ENCODER_PATH, 'rb'))
    classes = encoder.classes_
except Exception as e:
    print(f'인코더 로드 실패: {e}')
    raise SystemExit

models = {}
for floor, path in FLOOR_MODELS.items():
    if os.path.exists(path):
        try:
            models[floor] = tf.keras.models.load_model(path)
            print(f'모델 로드 완료: {floor}')
        except Exception as e:
            print(f'{floor} 모델 로드 실패: {e}')
    else:
        print(f'모델 파일 없음: {path}')

print('\n' + '=' * 80)
print('LSTM 액체 감지 예측기 - 시각화'.center(80))
print('=' * 80 + '\n')


def to_float(x):
    try:
        s = str(x).strip().replace('\ufeff', '')
        if s == '':
            return None
        return float(s)
    except Exception:
        return None


def find_floor_type(text: str):
    t = text.lower().strip().strip('"').strip("'")
    if not t:
        return None

    for floor, aliases in FLOOR_ALIASES.items():
        for a in aliases:
            if t == a.lower():
                return floor

    for floor, aliases in FLOOR_ALIASES.items():
        for a in aliases:
            al = a.lower()
            if al and (al in t or t in al):
                return floor

    return None


def read_new_format_csv(fp):
    final_base = None
    currents = []
    stored_changes = []
    row_bases = []
    timestamps = []

    for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
        try:
            with open(fp, 'r', encoding=enc, errors='ignore', newline='') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                names = [str(x).strip().replace('\ufeff', '') for x in reader.fieldnames]
                if 'record_type' not in names:
                    continue

                t = 0.1
                for row in reader:
                    rec = (row.get('record_type') or '').strip().lower()
                    if rec == 'final_base' and final_base is None:
                        final_base = to_float(row.get('base_raw'))
                    elif rec == 'data':
                        current_raw = to_float(row.get('current_raw'))
                        base_raw = to_float(row.get('base_raw'))
                        change = to_float(row.get('change'))
                        if current_raw is None:
                            continue
                        currents.append(current_raw)
                        row_bases.append(base_raw)
                        stored_changes.append(change)
                        timestamps.append(f'{t:.1f}s')
                        t += 0.1

                if len(currents) < WINDOW:
                    return None, None, None, None

                base = final_base
                if base is None:
                    valid_row_bases = [b for b in row_bases if b is not None]
                    if not valid_row_bases:
                        return None, None, None, None
                    base = float(np.mean(valid_row_bases))

                buffer = np.array(currents, dtype=np.float32)
                if RECALC_CHANGE_FROM_BASELINE:
                    changes = np.abs(buffer - base).astype(np.float32)
                else:
                    fixed_changes = []
                    for i, ch in enumerate(stored_changes):
                        if ch is None:
                            fallback_base = row_bases[i] if row_bases[i] is not None else base
                            ch = abs(buffer[i] - fallback_base)
                        fixed_changes.append(ch)
                    changes = np.array(fixed_changes, dtype=np.float32)
                return float(base), buffer, changes, timestamps
        except Exception:
            continue
    return None, None, None, None


def read_old_format_csv(fp):
    base = None
    data_points = []
    row_bases = []
    stored_changes = []
    timestamps = []

    for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
        try:
            with open(fp, 'r', encoding=enc, errors='ignore') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 2:
                        continue

                    if base is None and 'Final' in parts[0] and 'Base' in parts[1]:
                        for val in parts[4:]:
                            if val.strip():
                                maybe = to_float(val)
                                if maybe is not None:
                                    base = maybe
                                    break

                    if 'Time' in parts[0] and len(parts) >= 11:
                        current_raw = None
                        base_raw = None
                        change = None
                        timestamps.append(parts[1] if len(parts) > 1 else '')

                        for i, p in enumerate(parts):
                            if 'BaseRaw' in p and i + 1 < len(parts):
                                base_raw = to_float(parts[i + 1])
                            if 'CurrentRaw' in p and i + 1 < len(parts):
                                current_raw = to_float(parts[i + 1])
                            if 'Change' in p and i + 1 < len(parts):
                                change = to_float(parts[i + 1])

                        if current_raw is not None:
                            data_points.append(current_raw)
                            row_bases.append(base_raw)
                            stored_changes.append(change)

            if base is not None and len(data_points) >= WINDOW:
                buffer = np.array(data_points, dtype=np.float32)
                if RECALC_CHANGE_FROM_BASELINE:
                    changes = np.abs(buffer - base).astype(np.float32)
                else:
                    fixed_changes = []
                    for i, ch in enumerate(stored_changes):
                        fallback_base = row_bases[i] if row_bases[i] is not None else base
                        if ch is None:
                            ch = abs(buffer[i] - fallback_base)
                        fixed_changes.append(ch)
                    changes = np.array(fixed_changes, dtype=np.float32)
                return float(base), buffer, changes, timestamps
        except Exception:
            continue
    return None, None, None, None


def read_csv_with_fixed_baseline(fp):
    base, buffer, changes, timestamps = read_new_format_csv(fp)
    if base is not None:
        return base, buffer, changes, timestamps
    return read_old_format_csv(fp)


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

    change_raw = base_eff - recent_mean if floor_type == '검대' else recent_mean - base_eff
    direction = np.clip(change_raw / 300.0, -1.0, 1.0)
    direction_channel = np.full((WINDOW,), direction)

    window = np.stack([norm_sig, deriv1, deriv2, direction_channel])
    return window[np.newaxis, :, :]


def predict(buffer, changes, baseline, floor_type, model):
    predictions = []
    max_start = len(buffer) - WINDOW
    if max_start < 0:
        return []

    grad_thresh = FLOOR_THRESHOLDS.get(floor_type, {'gradient': 0.5})['gradient']
    change_gradient = np.gradient(changes)
    floor_idx = np.where(classes == '바닥')[0][0] if '바닥' in classes else -1

    for start_idx in range(0, max_start + 1, STEP):
        window = create_window(buffer, start_idx, baseline, floor_type)
        if window is None:
            continue

        window_changes = changes[start_idx:start_idx + WINDOW]
        window_gradient = change_gradient[start_idx:start_idx + WINDOW]
        avg_change = np.mean(window_changes)
        max_change = np.max(window_changes)
        max_gradient = np.max(np.abs(window_gradient))

        if model:
            pred = model.predict(np.transpose(window, (0, 2, 1)), verbose=0)[0]
        else:
            pred = np.zeros(len(classes), dtype=np.float32)
            if floor_idx >= 0:
                pred[floor_idx] = 1.0

        orig_label = classes[np.argmax(pred)]
        orig_prob = pred.max() * 100

        if max_change > 40:
            final_label = orig_label
            final_prob = max(orig_prob, 90.0)
        elif avg_change > 25:
            final_label = orig_label
            final_prob = max(orig_prob, 85.0)
        elif max_gradient < grad_thresh and avg_change < 12:
            final_label = '바닥'
            final_prob = 95.0
        else:
            final_label = orig_label
            final_prob = orig_prob

        predictions.append({
            'start': start_idx,
            'end': start_idx + WINDOW,
            'label': final_label,
            'prob': final_prob,
            'orig_label': orig_label,
            'orig_prob': orig_prob,
            'max_change': max_change,
            'max_gradient': max_gradient,
        })

    return predictions


def visualize(predictions, buffer, changes, baseline, floor_type):
    FLOOR_COLOR = '#E8E8E8'
    LIQUID_COLOR = '#5C2F0F'

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.6, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1_twin = ax1.twinx()
    ax1.plot(buffer, 'b-', lw=2, label='CurrentRaw')
    ax1.axhline(baseline, color='green', ls='--', lw=2, label=f'Base: {baseline:.1f}')
    ax1_twin.plot(changes, 'r:', alpha=0.7, lw=1.8, label='Change')

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
    ax2.bar(range(len(predictions)), [r['prob'] for r in predictions],
            color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_ylim(0, 105)
    ax2.set_title('신뢰도')
    ax2.grid(alpha=0.3, axis='y')

    ax3 = fig.add_subplot(gs[2])
    ax3.plot([r['max_gradient'] for r in predictions], 'purple', alpha=0.8)
    ax3.axhline(FLOOR_THRESHOLDS[floor_type]['gradient'], color='red', ls='--', lw=2)
    ax3.set_title('Gradient (최대 변화율)')

    ax4 = fig.add_subplot(gs[3])
    for i, r in enumerate(predictions):
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        ax4.add_patch(Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='k', lw=1.2))
        txt = '바' if r['label'] == '바닥' else '액'
        ax4.text(i + 0.5, 0.5, txt, ha='center', va='center', fontweight='bold',
                 fontsize=10 if len(predictions) < 50 else 8, color='white')

    ax4.set_xlim(0, len(predictions))
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.set_title('예측 시퀀스')

    legend_handles = [
        Patch(facecolor=FLOOR_COLOR, edgecolor='black', label='바닥'),
        Patch(facecolor=LIQUID_COLOR, edgecolor='black', label='액체'),
    ]
    ax4.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)

    plt.tight_layout()
    plt.show()


while True:
    print('\n' + '─' * 80)
    print(f"지원 바닥: {', '.join(models.keys()) if models else '없음'} | 종료: q")
    print('─' * 80)

    cmd = input('바닥 타입 ▶ ').strip()
    if cmd.lower() in ['q', 'quit', 'exit', 'ㅂㅂ', '']:
        print('\n예측기 종료!')
        break

    floor_type = find_floor_type(cmd)
    if not floor_type or floor_type not in models:
        print('지원되지 않는 바닥 타입입니다.')
        continue

    print(f'선택: {floor_type}')

    path = input('CSV 경로 ▶ ').strip().strip('"')
    if not os.path.exists(path):
        print('파일이 없습니다.')
        continue

    base, buffer, changes, ts = read_csv_with_fixed_baseline(path)
    if base is None:
        print('CSV 파싱 실패')
        continue

    print(f'\n파일: {os.path.basename(path)}')
    print(f'Base: {base:.1f} | 길이: {len(buffer)}')

    predictions = predict(buffer, changes, base, floor_type, models.get(floor_type))
    print(f'예측 완료: {len(predictions)}개 구간')

    cnt = Counter(r['label'] for r in predictions)
    for label, count in cnt.most_common():
        perc = count / len(predictions) * 100
        print(f'  {label}: {count}개 ({perc:.1f}%)')

    visualize(predictions, buffer, changes, base, floor_type)
    print('시각화 완료!\n')
