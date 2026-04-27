import numpy as np
import os
import glob
import csv
from pathlib import Path
import pickle
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = r"C:\Users\MASL\Desktop\코드개선\3차액체중간포함\액체중간포함"
SAVE_DIR = "./kumoh_lstm_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 15
CHANGE_THRESHOLD = 5.5
LOW_CHANGE_THRESHOLD = 4.5
PLATEAU_MEAN_THRESHOLD = 7.0
PLATEAU_MEDIAN_THRESHOLD = 6.0
SIGN_CONSISTENCY_THRESHOLD = 0.88
NOISE_RATIO_THRESHOLD = 0.82
SPIKE_RATIO_THRESHOLD = 2.20
SMOOTH_K = 5

LIQUIDS = ['바닥', '물', '말차', '커피', '콜라', '토마토', '우유', '망고', '기름', '수박']
LIQUID_ALIASES = {
    'water': '물', 'coffee': '커피', 'cola': '콜라', 'milk': '우유',
    'mango': '망고', 'oil': '기름', 'matcha': '말차', 'tomato': '토마토', 'watermelon': '수박'
}
FLOOR_KEYWORDS = {
    '검대': ['검대'], '회대': ['회대'], 
    '황대': ['황대'],
    '207회바': ['greyfloor', '회색바닥'],
    '흰책상': ['white', '하양', '흰', 'whitedesk'],
    '나타': ['나타'], '회타': ['회타'],
}
REVERSE_DIRECTION_FLOORS = {'검대'}


# zip에서 잘라둔 liquid 구간 파일 대응용
LIQUID_SEGMENT_KEY = '_liquid_'
FLOOR_SEGMENT_KEY = '_floor_'


def iter_csv_files(data_dir):
    patterns = [
        os.path.join(data_dir, '*.csv'),
        os.path.join(data_dir, '**', '*.csv'),
    ]
    seen = set()
    for pattern in patterns:
        for fp in glob.glob(pattern, recursive=True):
            if fp not in seen and os.path.isfile(fp):
                seen.add(fp)
                yield fp


def is_liquid_segment_file(fp: str) -> bool:
    name = os.path.basename(fp).lower()
    parent = os.path.basename(os.path.dirname(fp)).lower()
    return (LIQUID_SEGMENT_KEY in name) or ('liquid' in parent and FLOOR_SEGMENT_KEY not in name)


def is_floor_segment_file(fp: str) -> bool:
    name = os.path.basename(fp).lower()
    parent = os.path.basename(os.path.dirname(fp)).lower()
    return (FLOOR_SEGMENT_KEY in name) or ('floor' in parent and LIQUID_SEGMENT_KEY not in name)


def detect_floor_type_from_path(fp: str):
    parts = [os.path.basename(fp), os.path.dirname(fp), str(Path(fp).parent)]
    for part in parts:
        floor = detect_floor_type(str(part))
        if floor:
            return floor
    return None


def detect_liquid_from_filename(filename: str):
    lower = filename.lower()
    for key, kor in LIQUID_ALIASES.items():
        if key in lower:
            return kor
    for kor in LIQUIDS:
        if kor != '바닥' and kor in filename:
            return kor
    return None


def to_float(x):
    try:
        return float(str(x).strip())
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


def read_csv_recalc_change(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    def parse_new_format(lines):
        baseline_values = []
        currents = []
        row_bases = []

        reader = csv.DictReader(lines)
        fields = [str(name).strip() for name in (reader.fieldnames or [])]
        if 'record_type' not in fields:
            return None, None, None

        for row in reader:
            record_type = str(row.get('record_type', '')).strip().lower()
            baseline_raw = to_float(row.get('baseline_raw', ''))
            base_raw = to_float(row.get('base_raw', ''))
            current_raw = to_float(row.get('current_raw', ''))

            if record_type == 'baseline':
                if baseline_raw is not None:
                    baseline_values.append(baseline_raw)
                elif base_raw is not None:
                    baseline_values.append(base_raw)
                continue

            if record_type == 'data' and current_raw is not None:
                currents.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)

        if len(currents) < WINDOW_SIZE:
            return None, None, None

        if baseline_values:
            base = float(np.mean(baseline_values))
        else:
            valid_row_bases = [b for b in row_bases if b is not None]
            if not valid_row_bases:
                return None, None, None
            base = float(np.mean(valid_row_bases))

        currents = np.array(currents, dtype=np.float32)
        changes = np.abs(currents - base).astype(np.float32)
        return base, currents, changes

    def parse_old_format(lines):
        baseline_values = []
        currents = []
        row_bases = []
        final_base = None

        for raw_line in lines:
            line = str(raw_line).strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(',')]
            if not parts:
                continue

            # 예:
            # Collecting baseline...,RawAvg,416.8
            if parts[0].startswith('Collecting baseline'):
                if len(parts) >= 3 and parts[1] == 'RawAvg':
                    v = to_float(parts[2])
                    if v is not None:
                        baseline_values.append(v)
                continue

            # 예:
            # Final,Base,Raw,Average,417.68
            if len(parts) >= 5 and parts[:4] == ['Final', 'Base', 'Raw', 'Average']:
                v = to_float(parts[4])
                if v is not None:
                    final_base = v
                continue

            # 예:
            # Time,5.0s,|,BaseRaw,417.7,|,CurrentRaw,427.8,|,Change,0.0,=>,Floor,(empty)
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
                    currents.append(current_raw)
                if base_raw is not None:
                    row_bases.append(base_raw)

        if len(currents) < WINDOW_SIZE:
            return None, None, None

        if final_base is not None:
            base = float(final_base)
        elif baseline_values:
            base = float(np.mean(baseline_values))
        else:
            valid_row_bases = [b for b in row_bases if b is not None]
            if not valid_row_bases:
                return None, None, None
            base = float(np.mean(valid_row_bases))

        currents = np.array(currents, dtype=np.float32)
        changes = np.abs(currents - base).astype(np.float32)
        return base, currents, changes

    for enc in encodings:
        try:
            with open(fp, 'r', encoding=enc, errors='ignore', newline='') as f:
                lines = f.readlines()

            # 1순위: 신형 형식
            base, currents, changes = parse_new_format(lines)
            if base is not None:
                return base, currents, changes

            # 2순위: 구형 로그 형식
            base, currents, changes = parse_old_format(lines)
            if base is not None:
                return base, currents, changes

        except Exception:
            continue

    return None, None, None


def get_floor_ref_params(floor_type):
    if floor_type in {'나타', '회타'}:
        return 0.020, 0.65, 5.0, 0.68, 1.85
    if floor_type == '황대':
        return 0.028, 0.75, 6.0, 0.82, 2.20
    return 0.026, 0.75, 6.0, 0.78, 2.05


def build_recent_floor_reference(signal, baseline, floor_type):
    alpha, ref_grad, ref_band, _, _ = get_floor_ref_params(floor_type)
    sig = smooth_signal(remove_spikes(signal))
    grad = np.gradient(sig)
    ref = np.empty_like(sig)
    ref[0] = baseline
    for i in range(1, len(sig)):
        stable_grad = abs(grad[i]) < ref_grad
        close_to_ref = abs(sig[i] - ref[i - 1]) < ref_band
        if stable_grad and close_to_ref:
            ref[i] = (1.0 - alpha) * ref[i - 1] + alpha * sig[i]
        else:
            ref[i] = ref[i - 1]
    return sig, ref


def create_window_with_label(signal, changes, floor_ref, start, base, floor_type, liquid):
    if start + WINDOW_SIZE > len(signal):
        return None, None

    sig = signal[start:start + WINDOW_SIZE]
    chg = changes[start:start + WINDOW_SIZE]
    ref_w = floor_ref[start:start + WINDOW_SIZE]
    diff_w = sig - ref_w

    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])

    base_eff = base if len(signal) < 20 else 0.7 * base + 0.3 * np.mean(signal[:20])
    ctx_start = max(0, start - 5)
    ctx_end = min(len(signal), start + WINDOW_SIZE + 5)
    ctx_mean = np.mean(signal[ctx_start:ctx_end])
    direction_raw = base_eff - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - base_eff
    direction = np.clip(direction_raw / 300.0, -1.0, 1.0)
    direction_ch = np.full(WINDOW_SIZE, direction)

    window = np.stack([norm, d1, d2, direction_ch])

    max_change = float(np.max(chg))
    mean_floor_diff = float(np.mean(np.abs(diff_w)))
    median_floor_diff = float(np.abs(np.median(diff_w)))
    sign_consistency = float(max(np.mean(diff_w >= 0), np.mean(diff_w <= 0)))
    noise_ratio, spike_ratio = compute_noise_metrics(diff_w, sig)
    _, _, _, floor_noise_th, floor_spike_th = get_floor_ref_params(floor_type)

    strong_liquid = max_change >= CHANGE_THRESHOLD
    plateau_liquid = (
        max_change >= LOW_CHANGE_THRESHOLD and
        mean_floor_diff >= PLATEAU_MEAN_THRESHOLD and
        median_floor_diff >= PLATEAU_MEDIAN_THRESHOLD and
        sign_consistency >= SIGN_CONSISTENCY_THRESHOLD and
        noise_ratio <= min(NOISE_RATIO_THRESHOLD, floor_noise_th) and
        spike_ratio <= min(SPIKE_RATIO_THRESHOLD, floor_spike_th)
    )
    liquid_flag = strong_liquid or plateau_liquid
    label = liquid if liquid_flag else '바닥'
    return window, label


def detect_floor_type(filename):
    lower = filename.lower()
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None


def train_models(data_dir=DATA_DIR, save_dir=SAVE_DIR):
    """
    - 기존 일반 csv도 그대로 읽음
    - zip에서 잘라둔 *_liquid_*.csv 는 전 구간을 액체 라벨로만 학습에 사용
    - *_floor_*.csv 는 학습에서 제외
    """
    encoder = LabelEncoder().fit(LIQUIDS)
    with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

    floor_files = {f: [] for f in FLOOR_KEYWORDS}
    for fp in iter_csv_files(data_dir):
        name = os.path.basename(fp).lower()

        # zip에서 잘라낸 floor 구간은 이번 비교군 학습에서 제외
        if is_floor_segment_file(fp):
            continue

        if any(skip in name for skip in ['바닥', 'base']):
            continue

        floor = detect_floor_type_from_path(fp)
        if floor:
            floor_files[floor].append(fp)

    for floor, file_list in floor_files.items():
        if not file_list:
            continue
        X_all, y_all = [], []
        for fp in file_list:
            liquid = detect_liquid_from_filename(os.path.basename(fp))
            if not liquid:
                continue

            base, signal, changes = read_csv_recalc_change(fp)
            if base is None:
                continue

            smooth_sig, floor_ref = build_recent_floor_reference(signal, base, floor)
            max_start = len(signal) - WINDOW_SIZE
            if max_start <= 0:
                continue

            step = max(1, max_start // 25)
            liquid_only_segment = is_liquid_segment_file(fp)

            for start in range(0, max_start + 1, step):
                if liquid_only_segment:
                    sig = smooth_sig[start:start + WINDOW_SIZE]
                    if len(sig) < WINDOW_SIZE:
                        continue

                    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
                    d1 = np.diff(norm, prepend=norm[0])
                    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
                    d2 = np.diff(d1, prepend=d1[0])

                    base_eff = base if len(signal) < 20 else 0.7 * base + 0.3 * np.mean(signal[:20])
                    ctx_start = max(0, start - 5)
                    ctx_end = min(len(signal), start + WINDOW_SIZE + 5)
                    ctx_mean = np.mean(signal[ctx_start:ctx_end])
                    direction_raw = base_eff - ctx_mean if floor in REVERSE_DIRECTION_FLOORS else ctx_mean - base_eff
                    direction = np.clip(direction_raw / 300.0, -1.0, 1.0)
                    direction_ch = np.full(WINDOW_SIZE, direction)

                    win = np.stack([norm, d1, d2, direction_ch])
                    label = liquid
                else:
                    win, label = create_window_with_label(smooth_sig, changes, floor_ref, start, base, floor, liquid)
                    if win is None:
                        continue

                for _ in range(2):
                    aug = win + np.random.normal(0, random.uniform(0.01, 0.04), win.shape)
                    aug[:3] *= random.uniform(0.95, 1.05)
                    X_all.append(aug)
                    y_all.append(label)

        if len(X_all) == 0:
            print(f'[SKIP] {floor}: 학습 데이터가 없습니다.')
            continue

        X = np.array(X_all).transpose(0, 2, 1)
        y = encoder.transform(y_all)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(LIQUIDS))

        classes_present = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, class_weights)}

        inp = Input(shape=(WINDOW_SIZE, 4))
        x = Conv1D(32, 3, padding='same', activation='relu')(inp)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64, return_sequences=False))(x)
        x = Dropout(0.3)(x)
        out = Dense(len(LIQUIDS), activation='softmax')(x)
        model = Model(inp, out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        save_path = os.path.join(save_dir, f'model_{floor}.keras')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        print(f'[TRAIN] 바닥={floor} | 파일수={len(file_list)} | 샘플수={len(X)}')
        model.fit(
            X, y_cat,
            validation_split=0.2,
            epochs=60,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )


if __name__ == '__main__':
    train_models()
