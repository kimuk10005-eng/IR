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
ENCODER_PATH = os.path.join(SAVE_DIR, 'encoder.pkl')
WINDOW = 15
STEP = 5

# 기존 구조는 유지하되, plateau(변화 사이 내부 액체) 유지용 임계값만 추가
FLOOR_THRESHOLDS = {
    '나무':   {'gradient': 0.5, 'change': 5, 'local_change': 4.0,
              'start_grad': 0.9, 'start_mean_local': 3.0, 'start_area_local': 45.0,
              'keep_mean_local': 2.0, 'keep_area_local': 30.0,
              'release_mean_local': 1.0, 'release_area_local': 16.0,
              'memory_windows': 2, 'release_windows': 2, 'model_liquid_prob': 72.0},
    '황대':   {'gradient': 0.5, 'change': 3, 'local_change': 3.0,
              'start_grad': 1.4, 'start_mean_local': 3.0, 'start_area_local': 45.0,
              'keep_mean_local': 1.8, 'keep_area_local': 28.0,
              'release_mean_local': 1.0, 'release_area_local': 16.0,
              'memory_windows': 2, 'release_windows': 2, 'model_liquid_prob': 72.0},
    '회대':   {'gradient': 0.5, 'change': 3, 'local_change': 3.0,
              'start_grad': 1.2, 'start_mean_local': 3.2, 'start_area_local': 48.0,
              'keep_mean_local': 2.0, 'keep_area_local': 32.0,
              'release_mean_local': 1.2, 'release_area_local': 18.0,
              'memory_windows': 2, 'release_windows': 2, 'model_liquid_prob': 72.0},
    '검대':   {'gradient': 0.5, 'change': 3, 'local_change': 3.0,
              'start_grad': 0.9, 'start_mean_local': 4.0, 'start_area_local': 60.0,
              'keep_mean_local': 2.6, 'keep_area_local': 38.0,
              'release_mean_local': 1.5, 'release_area_local': 22.0,
              'memory_windows': 2, 'release_windows': 2, 'model_liquid_prob': 72.0},
    '그마':   {'gradient': 0.5, 'change': 3, 'local_change': 3.0,
              'start_grad': 1.2, 'start_mean_local': 3.2, 'start_area_local': 48.0,
              'keep_mean_local': 2.0, 'keep_area_local': 32.0,
              'release_mean_local': 1.2, 'release_area_local': 18.0,
              'memory_windows': 2, 'release_windows': 2, 'model_liquid_prob': 72.0},
    '207회바': {'gradient': 0.5, 'change': 3, 'local_change': 3.0,
              'start_grad': 1.2, 'start_mean_local': 3.2, 'start_area_local': 48.0,
              'keep_mean_local': 2.0, 'keep_area_local': 32.0,
              'release_mean_local': 1.2, 'release_area_local': 18.0,
              'memory_windows': 2, 'release_windows': 2, 'model_liquid_prob': 72.0},
    '흰책상': {'gradient': 0.5, 'change': 3, 'local_change': 3.0,
              'start_grad': 1.1, 'start_mean_local': 3.0, 'start_area_local': 45.0,
              'keep_mean_local': 1.8, 'keep_area_local': 28.0,
              'release_mean_local': 1.0, 'release_area_local': 16.0,
              'memory_windows': 2, 'release_windows': 2, 'model_liquid_prob': 72.0},
    '나타':   {'gradient': 0.5, 'change': 3, 'local_change': 3.0,
              'start_grad': 1.0, 'start_mean_local': 3.0, 'start_area_local': 45.0,
              'keep_mean_local': 1.8, 'keep_area_local': 28.0,
              'release_mean_local': 1.0, 'release_area_local': 16.0,
              'memory_windows': 2, 'release_windows': 2, 'model_liquid_prob': 72.0},
    '회타':   {'gradient': 0.5, 'change': 3, 'local_change': 3.0,
              'start_grad': 1.2, 'start_mean_local': 3.2, 'start_area_local': 48.0,
              'keep_mean_local': 2.0, 'keep_area_local': 32.0,
              'release_mean_local': 1.2, 'release_area_local': 18.0,
              'memory_windows': 2, 'release_windows': 2, 'model_liquid_prob': 72.0},
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
    '회대': ['회대', '회색대리석', '회색', 'gray', '회'],
    '검대': ['검대', '검정색대리석', '검정', 'black', '검'],
    '그마': ['그마', 'greymarble'],
    '207회바': ['회바', '207', '207greyfloor', 'greyfloor'],
    '흰책상': ['흰책상', 'white', 'whitedesk'],
    '나타': ['나타'],
    '회타': ['회타']
}

REVERSE_DIRECTION_FLOORS = {'검대'}
LOCAL_REF_BACK = 20
LOCAL_REF_GUARD = 3
SMOOTH_KERNEL = 5

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

print("\n" + "=" * 80)
print("LSTM 액체 감지 예측기 - local baseline + hysteresis".center(80))
print("=" * 80 + "\n")


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


def smooth_signal(x, kernel=SMOOTH_KERNEL):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0 or kernel <= 1:
        return x.copy()
    kernel = min(kernel, len(x))
    if kernel % 2 == 0:
        kernel += 1
        kernel = min(kernel, len(x) if len(x) % 2 == 1 else max(1, len(x) - 1))
    if kernel <= 1:
        return x.copy()
    pad = kernel // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    w = np.ones(kernel, dtype=np.float32) / kernel
    return np.convolve(xp, w, mode='valid').astype(np.float32)


def compute_local_reference(signal, start_idx, global_base, back=LOCAL_REF_BACK, guard=LOCAL_REF_GUARD):
    ref_end = max(0, start_idx - guard)
    ref_start = max(0, ref_end - back)
    ref = signal[ref_start:ref_end]
    if len(ref) < 5:
        return float(global_base)

    ref_smooth = smooth_signal(ref, kernel=min(5, len(ref)))
    ref_grad = np.abs(np.gradient(ref_smooth)) if len(ref_smooth) >= 2 else np.zeros_like(ref_smooth)
    stable_mask = ref_grad <= np.percentile(ref_grad, 70)
    stable_ref = ref_smooth[stable_mask]
    if len(stable_ref) < 3:
        stable_ref = ref_smooth
    return float(np.mean(stable_ref))


def read_csv_with_fixed_baseline(fp):
    baseline_values = []
    row_bases = []
    data_points = []
    timestamps = []

    for enc in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
        try:
            with open(fp, 'r', encoding=enc, errors='ignore', newline='') as f:
                reader = csv.DictReader(f)
                fields = [name.strip() for name in (reader.fieldnames or [])]
                if 'record_type' not in fields:
                    continue

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

            if len(data_points) >= WINDOW:
                if baseline_values:
                    base = float(np.mean(baseline_values))
                elif row_bases:
                    base = float(np.mean(row_bases))
                else:
                    continue

                buffer = np.array(data_points, dtype=np.float32)
                global_changes = np.abs(buffer - base).astype(np.float32)
                return base, buffer, global_changes, timestamps
        except Exception:
            continue

    return None, None, None, None


def create_window(full_signal, start_idx, baseline, floor_type):
    if start_idx + WINDOW > len(full_signal):
        return None

    sig = full_signal[start_idx:start_idx + WINDOW]
    sig_smooth = smooth_signal(sig, kernel=min(SMOOTH_KERNEL, len(sig)))

    norm_sig = (sig_smooth - sig_smooth.mean()) / (sig_smooth.std() + 1e-8)
    deriv1 = np.diff(norm_sig, prepend=norm_sig[0])
    deriv1 = np.convolve(deriv1, np.ones(3, dtype=np.float32) / 3, mode='same')
    deriv2 = np.diff(deriv1, prepend=deriv1[0])

    local_ref = compute_local_reference(full_signal, start_idx, baseline)
    if floor_type in REVERSE_DIRECTION_FLOORS:
        dynamic_ch = np.clip((local_ref - sig_smooth) / 40.0, -1.0, 1.0)
    else:
        dynamic_ch = np.clip((sig_smooth - local_ref) / 40.0, -1.0, 1.0)

    window = np.stack([norm_sig, deriv1, deriv2, dynamic_ch], axis=0)
    return window[np.newaxis, :, :]


def best_non_floor_label(pred_vector):
    if '바닥' not in classes:
        return classes[int(np.argmax(pred_vector))]
    floor_idx = int(np.where(classes == '바닥')[0][0])
    tmp = pred_vector.copy()
    tmp[floor_idx] = -1.0
    return classes[int(np.argmax(tmp))]


def summarize_window_state(r, params):
    strong_boundary = (r['max_gradient'] >= params['start_grad'])
    strong_offset = (r['mean_local_change'] >= params['start_mean_local']) or (r['area_local_change'] >= params['start_area_local'])
    keep_offset = (r['mean_local_change'] >= params['keep_mean_local']) or (r['area_local_change'] >= params['keep_area_local'])
    release_ready = (r['mean_local_change'] <= params['release_mean_local']) and (r['area_local_change'] <= params['release_area_local']) and (r['max_gradient'] < params['start_grad'] * 0.7)
    model_liquid = (r['orig_label'] != '바닥') and (r['orig_prob'] >= params['model_liquid_prob'])
    return strong_boundary, strong_offset, keep_offset, release_ready, model_liquid


def apply_hysteresis_postprocess(predictions, floor_type):
    if not predictions:
        return predictions

    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['황대'])
    state = 0
    memory = 0
    release_count = 0
    labels = []
    liquid_labels = []

    for r in predictions:
        strong_boundary, strong_offset, keep_offset, release_ready, model_liquid = summarize_window_state(r, params)
        start_trigger = strong_boundary or strong_offset or (model_liquid and r['mean_local_change'] >= params['keep_mean_local'] * 0.8)
        keep_trigger = keep_offset or (memory > 0 and r['mean_local_change'] >= params['release_mean_local']) or model_liquid

        if state == 0:
            if start_trigger:
                state = 1
                memory = params['memory_windows']
                release_count = 0
        else:
            if strong_boundary:
                memory = params['memory_windows']
            else:
                memory = max(0, memory - 1)

            if keep_trigger:
                release_count = 0
            elif release_ready:
                release_count += 1
                if release_count >= params['release_windows']:
                    state = 0
                    memory = 0
                    release_count = 0
            else:
                release_count = 0

        labels.append(state)
        liquid_labels.append(r['fallback_liquid_label'])

    for i in range(1, len(labels) - 1):
        if labels[i] == 0 and labels[i - 1] == 1 and labels[i + 1] == 1:
            labels[i] = 1

    for i, lab in enumerate(labels):
        if lab == 0:
            predictions[i]['label'] = '바닥'
            predictions[i]['prob'] = max(predictions[i]['prob'], 88.0)
        else:
            if predictions[i]['label'] == '바닥' or predictions[i]['prob'] < 75.0:
                predictions[i]['label'] = liquid_labels[i]
                predictions[i]['prob'] = max(predictions[i]['prob'], 78.0)

    return predictions


def predict(buffer, global_changes, baseline, floor_type, model):
    predictions = []
    max_start = len(buffer) - WINDOW
    if max_start < 0:
        return []

    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['황대'])
    grad_thresh = params['gradient']
    global_change_thresh = params['change']
    local_change_thresh = params['local_change']

    smoothed = smooth_signal(buffer)
    full_grad = np.abs(np.gradient(smoothed)) if len(smoothed) >= 2 else np.zeros_like(smoothed)
    floor_idx = np.where(classes == '바닥')[0][0] if '바닥' in classes else -1

    for start_idx in range(0, max_start + 1, STEP):
        end_idx = start_idx + WINDOW
        window = create_window(buffer, start_idx, baseline, floor_type)
        if window is None:
            continue

        sig = buffer[start_idx:end_idx]
        local_ref = compute_local_reference(buffer, start_idx, baseline)
        local_diff = sig - local_ref
        local_changes = np.abs(local_diff)
        global_window_changes = global_changes[start_idx:end_idx]
        window_grad = full_grad[start_idx:end_idx]

        mean_local_change = float(np.mean(local_changes))
        max_local_change = float(np.max(local_changes))
        area_local_change = float(np.sum(local_changes))
        std_local_diff = float(np.std(local_diff))
        avg_global_change = float(np.mean(global_window_changes))
        max_global_change = float(np.max(global_window_changes))
        max_gradient = float(np.max(window_grad))

        if model:
            pred = model.predict(np.transpose(window, (0, 2, 1)), verbose=0)[0]
        else:
            pred = np.zeros(len(classes), dtype=np.float32)
            if floor_idx >= 0:
                pred[floor_idx] = 1.0
            else:
                pred[0] = 1.0

        orig_label = classes[int(np.argmax(pred))]
        orig_prob = float(np.max(pred) * 100.0)
        fallback_liquid = best_non_floor_label(pred)

        strong_global = (max_global_change >= max(40.0, global_change_thresh * 8.0)) or (avg_global_change >= 25.0)
        calm_floor = (max_gradient < grad_thresh * 0.8) and (mean_local_change < local_change_thresh * 0.8)
        event_score = (
            max_gradient / (params['start_grad'] + 1e-8)
            + mean_local_change / (params['start_mean_local'] + 1e-8)
            + area_local_change / (params['start_area_local'] + 1e-8)
        )

        if strong_global:
            final_label = orig_label
            final_prob = max(orig_prob, 88.0)
        elif calm_floor and orig_prob < 75.0:
            final_label = '바닥'
            final_prob = 93.0
        else:
            final_label = orig_label
            final_prob = orig_prob

        predictions.append({
            'start': start_idx,
            'end': end_idx,
            'label': final_label,
            'prob': final_prob,
            'orig_label': orig_label,
            'orig_prob': orig_prob,
            'max_global_change': max_global_change,
            'avg_global_change': avg_global_change,
            'mean_local_change': mean_local_change,
            'max_local_change': max_local_change,
            'area_local_change': area_local_change,
            'std_local_diff': std_local_diff,
            'max_gradient': max_gradient,
            'local_ref': local_ref,
            'event_score': event_score,
            'fallback_liquid_label': fallback_liquid,
        })

    return apply_hysteresis_postprocess(predictions, floor_type)


def visualize(predictions, buffer, global_changes, baseline, floor_type):
    FLOOR_COLOR = '#E8E8E8'
    LIQUID_COLOR = '#5C2F0F'

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.6, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1_twin = ax1.twinx()
    ax1.plot(buffer, 'b-', lw=2, label='CurrentRaw')
    ax1.axhline(baseline, color='green', ls='--', lw=2, label=f'Base: {baseline:.1f}')
    ax1_twin.plot(global_changes, 'r:', alpha=0.7, lw=1.8, label='Global Change')

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
    ax3.axhline(FLOOR_THRESHOLDS[floor_type]['start_grad'], color='red', ls='--', lw=2)
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
        Patch(facecolor=LIQUID_COLOR, edgecolor='black', label='액체')
    ]
    ax4.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.show()


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

    base, buffer, global_changes, ts = read_csv_with_fixed_baseline(path)
    if base is None:
        print('CSV 파싱 실패')
        continue

    print(f'\n파일: {os.path.basename(path)}')
    print(f'Base: {base:.1f} | 길이: {len(buffer)}')

    predictions = predict(buffer, global_changes, base, floor_type, models[floor_type])
    print(f'예측 완료: {len(predictions)}개 구간')

    cnt = Counter('바닥' if r['label'] == '바닥' else '액체' for r in predictions)
    for label, count in cnt.most_common():
        perc = count / len(predictions) * 100
        print(f'  {label}: {count}개 ({perc:.1f}%)')

    visualize(predictions, buffer, global_changes, base, floor_type)
    print('시각화 완료!\n')
