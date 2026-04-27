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
SMOOTH_K = 5

FLOOR_THRESHOLDS = {
    '나무': {'gradient': 0.7, 'change_low': 6, 'change_strong': 18, 'plateau_mean': 8, 'plateau_med': 6, 'clear_mean': 3.0},
    '황대': {'gradient': 0.6, 'change_low': 5, 'change_strong': 15, 'plateau_mean': 6, 'plateau_med': 5, 'clear_mean': 2.5},
    '회대': {'gradient': 0.7, 'change_low': 6, 'change_strong': 18, 'plateau_mean': 8, 'plateau_med': 6, 'clear_mean': 3.0},
    '검대': {'gradient': 0.7, 'change_low': 6, 'change_strong': 18, 'plateau_mean': 8, 'plateau_med': 6, 'clear_mean': 3.0},
    '그마': {'gradient': 0.7, 'change_low': 6, 'change_strong': 18, 'plateau_mean': 8, 'plateau_med': 6, 'clear_mean': 3.0},
    '207회바': {'gradient': 0.7, 'change_low': 6, 'change_strong': 18, 'plateau_mean': 8, 'plateau_med': 6, 'clear_mean': 3.0},
    '흰책상': {'gradient': 0.7, 'change_low': 6, 'change_strong': 18, 'plateau_mean': 8, 'plateau_med': 6, 'clear_mean': 3.0},
    '나타': {'gradient': 0.7, 'change_low': 6, 'change_strong': 18, 'plateau_mean': 8, 'plateau_med': 6, 'clear_mean': 3.0},
    '회타': {'gradient': 0.7, 'change_low': 6, 'change_strong': 18, 'plateau_mean': 8, 'plateau_med': 6, 'clear_mean': 3.0},
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
                changes = np.abs(buffer - base).astype(np.float32)
                return base, buffer, changes, timestamps
        except Exception:
            continue

    return None, None, None, None


def build_recent_floor_reference(signal, baseline):
    sig = smooth_signal(signal)
    grad = np.gradient(sig)
    ref = np.empty_like(sig)
    ref[0] = baseline
    alpha = 0.04
    for i in range(1, len(sig)):
        stable_grad = abs(grad[i]) < 1.0
        close_to_ref = abs(sig[i] - ref[i - 1]) < 8.0
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


def predict(buffer, changes, baseline, floor_type, model):
    predictions = []
    max_start = len(buffer) - WINDOW
    if max_start < 0:
        return []

    params = FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])
    sig_smooth, change_gradient, floor_ref = build_recent_floor_reference(buffer, baseline)
    floor_idx = np.where(classes == '바닥')[0][0] if '바닥' in classes else -1

    for start_idx in range(0, max_start + 1, STEP):
        window = create_window(buffer, start_idx, baseline, floor_type)
        if window is None:
            continue

        raw_w = buffer[start_idx:start_idx + WINDOW]
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

        if model:
            pred = model.predict(np.transpose(window, (0, 2, 1)), verbose=0)[0]
        else:
            pred = np.zeros(len(classes), dtype=np.float32)
            pred[floor_idx] = 1.0

        orig_idx = int(np.argmax(pred))
        orig_label = classes[orig_idx]
        orig_prob = float(pred[orig_idx] * 100.0)
        best_non_floor_label, best_non_floor_prob = get_best_non_floor_label(pred)

        # 1차: 기존 simple change를 최대한 유지
        simple_liquid = max_change >= params['change_low']
        strong_liquid = max_change >= params['change_strong']
        hard_floor = (max_change < params['change_low'] * 0.8 and
                      mean_floor_diff < params['clear_mean'] and
                      max_gradient < params['gradient'])

        # 2차: 아주 조금만 추가하는 plateau 보정
        plateau_flag = (
            mean_floor_diff >= params['plateau_mean'] and
            median_floor_diff >= params['plateau_med'] and
            sign_consistency >= 0.80
        )

        if hard_floor:
            final_label = '바닥'
            final_prob = 94.0
        elif strong_liquid:
            if orig_label != '바닥':
                final_label = orig_label
                final_prob = max(orig_prob, 88.0)
            elif plateau_flag:
                final_label = best_non_floor_label
                final_prob = max(best_non_floor_prob, 80.0)
            else:
                final_label = '바닥'
                final_prob = 70.0
        elif simple_liquid and plateau_flag:
            if orig_label != '바닥':
                final_label = orig_label
                final_prob = max(orig_prob, 78.0)
            else:
                final_label = best_non_floor_label
                final_prob = max(best_non_floor_prob, 72.0)
        elif simple_liquid and orig_label != '바닥' and orig_prob >= 60.0:
            final_label = orig_label
            final_prob = orig_prob
        else:
            final_label = '바닥'
            final_prob = max(60.0, 100.0 - avg_change)

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
            'mean_floor_diff': mean_floor_diff,
            'median_floor_diff': median_floor_diff,
            'sign_consistency': sign_consistency,
        })

    return predictions


FLOOR_COLOR = '#e0e0e0'
LIQUID_COLOR = '#7b4a1e'

def visualize(predictions, buffer, changes, baseline, floor_type):
    if not predictions:
        print('시각화할 예측 결과가 없습니다.')
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.0, 2.2, 1.4, 1.8], hspace=0.30)
    fig.suptitle(f'{floor_type} 바닥 - LSTM 실시간 액체 감지 결과', fontsize=20, fontweight='bold', y=0.97)

    # 1. 원신호 + 예측 배경 + change
    ax1 = fig.add_subplot(gs[0])
    for i, r in enumerate(predictions):
        color = FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR
        ax1.axvspan(r['start'], r['end'], color=color, alpha=0.22)

    ax1.plot(buffer, color='blue', lw=2, label='CurrentRaw')
    ax1.axhline(baseline, color='green', lw=2, ls='--', label=f'Base: {baseline:.1f}')
    ax1.set_title('')
    ax1.set_ylabel('')
    ax1.grid(alpha=0.25)
    ax1.legend(loc='upper left', fontsize=11)

    ax1r = ax1.twinx()
    ax1r.plot(changes, color='red', lw=2, ls=':', alpha=0.75, label='Global Change')
    ax1r.legend(loc='upper right', fontsize=11)

    # 2. 신뢰도 바
    ax2 = fig.add_subplot(gs[1])
    probs = [r['prob'] for r in predictions]
    colors = [FLOOR_COLOR if r['label'] == '바닥' else LIQUID_COLOR for r in predictions]
    ax2.bar(range(len(predictions)), probs, color=colors, edgecolor='black', linewidth=0.4)
    ax2.set_ylim(0, 105)
    ax2.set_title('신뢰도')
    ax2.grid(axis='y', alpha=0.25)

    # 3. Gradient
    ax3 = fig.add_subplot(gs[2])
    grads = [r['max_gradient'] for r in predictions]
    ax3.plot(grads, color='purple', alpha=0.85)
    ax3.axhline(FLOOR_THRESHOLDS.get(floor_type, FLOOR_THRESHOLDS['회대'])['gradient'], color='red', ls='--', lw=2)
    ax3.set_title('Gradient (최대 변화율)')
    ax3.grid(alpha=0.25)

    # 4. 예측 시퀀스
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
    ax4.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)

    plt.tight_layout()
    plt.show()


def main():
    print('\n' + '=' * 80)
    print('LSTM 액체 감지 예측기 - refined'.center(80))
    print('=' * 80 + '\n')

    while True:
        print('\n' + '─' * 80)
        print(f"지원 바닥: {', '.join(models.keys()) if models else '없음'} | 종료: q")
        print('─' * 80)

        cmd = input('바닥 타입 ▶ ').strip()
        if cmd.lower() in ['q', 'quit', 'exit', 'ㅂㅂ', '']:
            print('\n예측기 종료!')
            break

        floor_type = find_floor_type(cmd)
        if not floor_type:
            print('지원되지 않는 바닥 타입입니다.')
            continue
        if floor_type not in models:
            print(f'해당 바닥 모델이 로드되지 않았습니다: {floor_type}')
            continue

        csv_path = input('CSV 경로 ▶ ').strip().strip('"')
        if not csv_path:
            continue
        if not os.path.exists(csv_path):
            print(f'파일을 찾을 수 없습니다: {csv_path}')
            continue

        try:
            baseline, buffer, changes, times = read_csv_with_fixed_baseline(csv_path)
            if buffer is None or len(buffer) < WINDOW:
                print(f'데이터가 너무 짧습니다. 최소 {WINDOW}개 샘플 필요')
                continue

            preds = predict(buffer, changes, baseline, floor_type, models[floor_type])
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
