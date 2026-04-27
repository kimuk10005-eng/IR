import numpy as np
import os
import glob
import csv
import pickle
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, MaxPooling1D, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# 기존 경로/저장 구조 유지
DATA_DIR = r"C:\Users\MASL\Desktop\코드개선\3차_기존코드활용\3rd_db"
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

# 기존 다중 액체 대신 binary로 단순화
CLASSES = ['바닥', '액체']

LIQUID_ALIASES = {
    'water': '물', 'coffee': '커피', 'cola': '콜라', 'milk': '우유',
    'mango': '망고', 'oil': '기름', 'matcha': '말차', 'tomato': '토마토', 'watermelon': '수박'
}
FLOOR_KEYWORDS = {
    '나무': ['나무', 'wood'],
    '검대': ['검대', '검정색대리석', 'black'],
    '회대': ['회대', '회색대리석', 'gray', 'grey'],
    '황대': ['황대', '황색대리석', 'yellow'],
    '207회바': ['207회바', 'greyfloor', '회색바닥', '207'],
    '흰책상': ['흰책상', 'white', '하양', '흰', 'whitedesk'],
    '나타': ['나타'],
    '회타': ['회타'],
}
REVERSE_DIRECTION_FLOORS = {'검대'}


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
        if abs(float(sig[i]) - local_med) > k * local_std and \
           abs(float(sig[i]) - float(sig[i - 1])) > 1.2 * local_std and \
           abs(float(sig[i]) - float(sig[i + 1])) > 1.2 * local_std:
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
            base, currents, changes = parse_new_format(lines)
            if base is not None:
                return base, currents, changes
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


def detect_floor_type(filename):
    lower = filename.lower()
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None


def build_feature_window(sig, base, floor_type, global_signal, start_hint=0):
    sig = np.asarray(sig, dtype=np.float32)
    if len(sig) < 2:
        return None
    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])

    base_eff = base if len(global_signal) < 20 else 0.7 * base + 0.3 * np.mean(global_signal[:20])
    ctx_start = max(0, start_hint - 5)
    ctx_end = min(len(global_signal), start_hint + len(sig) + 5)
    ctx_mean = np.mean(global_signal[ctx_start:ctx_end])
    direction_raw = base_eff - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - base_eff
    direction = np.clip(direction_raw / 300.0, -1.0, 1.0)
    direction_ch = np.full(len(sig), direction, dtype=np.float32)

    window = np.stack([norm, d1, d2, direction_ch], axis=1)
    return window.astype(np.float32)


def resample_to_window(segment, target_len=WINDOW_SIZE):
    segment = np.asarray(segment, dtype=np.float32)
    if len(segment) == target_len:
        return segment
    if len(segment) == 1:
        return np.full(target_len, float(segment[0]), dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, len(segment))
    new_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_x, old_x, segment).astype(np.float32)


def extract_liquid_segments(signal, changes, floor_ref, floor_type):
    diff = signal - floor_ref
    abs_diff = np.abs(diff)
    alpha, ref_grad, ref_band, floor_noise_th, floor_spike_th = get_floor_ref_params(floor_type)
    flags = []
    for i in range(len(signal)):
        ch = float(changes[i])
        local_l = max(0, i - 2)
        local_r = min(len(signal), i + 3)
        diff_w = diff[local_l:local_r]
        sig_w = signal[local_l:local_r]
        mean_floor_diff = float(np.mean(np.abs(diff_w)))
        median_floor_diff = float(np.abs(np.median(diff_w)))
        sign_consistency = float(max(np.mean(diff_w >= 0), np.mean(diff_w <= 0))) if len(diff_w) else 0.0
        noise_ratio, spike_ratio = compute_noise_metrics(diff_w, sig_w)
        flag = (
            ch >= CHANGE_THRESHOLD or
            (
                ch >= LOW_CHANGE_THRESHOLD and
                mean_floor_diff >= PLATEAU_MEAN_THRESHOLD and
                median_floor_diff >= PLATEAU_MEDIAN_THRESHOLD and
                sign_consistency >= SIGN_CONSISTENCY_THRESHOLD and
                noise_ratio <= min(NOISE_RATIO_THRESHOLD, floor_noise_th) and
                spike_ratio <= min(SPIKE_RATIO_THRESHOLD, floor_spike_th)
            )
        )
        flags.append(flag)

    segments = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        s = i
        while i + 1 < n and flags[i + 1]:
            i += 1
        e = i
        if e - s + 1 >= max(3, WINDOW_SIZE // 5):
            segments.append([s, e])
        i += 1

    if not segments:
        return []

    merged = [segments[0]]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= 5:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    expanded = []
    for s, e in merged:
        s2 = max(0, s - 2)
        e2 = min(len(signal) - 1, e + 2)
        expanded.append((s2, e2))
    return expanded


def collect_positive_windows(signal, base, floor_ref, floor_type, segments):
    X, y = [], []
    for seg_start, seg_end in segments:
        seg = signal[seg_start:seg_end + 1]
        seg_len = len(seg)

        # 1) 액체구간 자체를 바로 반영 (짧으면 보간)
        core = resample_to_window(seg, WINDOW_SIZE)
        feat = build_feature_window(core, base, floor_type, signal, seg_start)
        if feat is not None:
            X.append(feat)
            y.append('액체')

        # 2) 좌우 문맥 포함 버전
        pad = max(2, (WINDOW_SIZE - min(seg_len, WINDOW_SIZE)) // 2)
        s = max(0, seg_start - pad)
        e = min(len(signal), seg_end + pad + 1)
        context_seg = signal[s:e]
        context_seg = resample_to_window(context_seg, WINDOW_SIZE)
        feat = build_feature_window(context_seg, base, floor_type, signal, s)
        if feat is not None:
            X.append(feat)
            y.append('액체')

        # 3) 중심 이동 버전 몇 개
        center = (seg_start + seg_end) // 2
        for shift in (-3, 0, 3):
            c = int(np.clip(center + shift, 0, len(signal) - 1))
            s = max(0, c - WINDOW_SIZE // 2)
            e = min(len(signal), s + WINDOW_SIZE)
            s = max(0, e - WINDOW_SIZE)
            local = signal[s:e]
            local = resample_to_window(local, WINDOW_SIZE)
            feat = build_feature_window(local, base, floor_type, signal, s)
            if feat is not None:
                X.append(feat)
                y.append('액체')
    return X, y


def collect_floor_windows(signal, changes, base, floor_ref, floor_type, segments):
    mask = np.zeros(len(signal), dtype=bool)
    for s, e in segments:
        left = max(0, s - 4)
        right = min(len(signal), e + 5)
        mask[left:right] = True

    X, y = [], []
    max_start = len(signal) - WINDOW_SIZE
    if max_start < 0:
        return X, y

    step = max(1, WINDOW_SIZE // 3)
    for start in range(0, max_start + 1, step):
        end = start + WINDOW_SIZE
        if np.any(mask[start:end]):
            continue
        chg = changes[start:end]
        if float(np.max(chg)) >= LOW_CHANGE_THRESHOLD:
            continue
        local = signal[start:end]
        feat = build_feature_window(local, base, floor_type, signal, start)
        if feat is not None:
            X.append(feat)
            y.append('바닥')
    return X, y


def augment_feature(feat):
    aug = feat.copy()
    aug[:, :3] += np.random.normal(0, random.uniform(0.005, 0.03), aug[:, :3].shape)
    aug[:, :3] *= random.uniform(0.97, 1.03)
    return aug.astype(np.float32)


def train_models(data_dir=DATA_DIR, save_dir=SAVE_DIR):
    encoder = LabelEncoder().fit(CLASSES)
    with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)

    floor_files = {f: [] for f in FLOOR_KEYWORDS}
    all_csv = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    print(f'[경로 확인] 학습 DB 폴더: {data_dir}')
    print(f'[경로 확인] 모델 저장 폴더: {save_dir}')
    print(f'[검색 결과] CSV 파일 수: {len(all_csv)}')
    if len(all_csv) == 0:
        print('[중단] 학습 폴더에서 CSV를 찾지 못했습니다. 하위 폴더 포함 경로를 확인하세요.')
        return

    unmatched = []
    for fp in all_csv:
        name = os.path.basename(fp).lower()
        if any(skip in name for skip in ['바닥', 'base']):
            continue
        floor = detect_floor_type(name)
        if floor:
            floor_files[floor].append(fp)
        else:
            unmatched.append(os.path.basename(fp))

    if unmatched:
        print(f'[참고] 바닥 키워드를 못 찾은 파일: {len(unmatched)}개')
        for name in unmatched[:15]:
            print(f'  - {name}')
        if len(unmatched) > 15:
            print('  ...')

    print('[바닥별 파일 수]')
    for floor_name, files in floor_files.items():
        print(f'  {floor_name}: {len(files)}개')

    trained_any = False

    for floor, file_list in floor_files.items():
        if not file_list:
            continue

        X_all, y_all = [], []
        used_files = 0
        used_segments = 0

        for fp in file_list:
            base, raw_signal, changes = read_csv_recalc_change(fp)
            if base is None:
                print(f'[SKIP] 읽기 실패: {os.path.basename(fp)}')
                continue

            smooth_sig, floor_ref = build_recent_floor_reference(raw_signal, base, floor)
            segments = extract_liquid_segments(smooth_sig, changes, floor_ref, floor)
            if not segments:
                print(f'[SKIP] 액체구간 없음: {os.path.basename(fp)}')
                continue

            pos_X, pos_y = collect_positive_windows(smooth_sig, base, floor_ref, floor, segments)
            neg_X, neg_y = collect_floor_windows(smooth_sig, changes, base, floor_ref, floor, segments)

            if not pos_X:
                print(f'[SKIP] 양성 윈도우 없음: {os.path.basename(fp)}')
                continue

            used_files += 1
            used_segments += len(segments)

            for feat, label in zip(pos_X + neg_X, pos_y + neg_y):
                X_all.append(feat)
                y_all.append(label)
                # 액체 샘플은 조금 더 증강
                repeat = 2 if label == '액체' else 1
                for _ in range(repeat):
                    X_all.append(augment_feature(feat))
                    y_all.append(label)

        if len(X_all) == 0:
            print(f'[SKIP] {floor}: 학습 데이터가 없습니다.')
            continue

        X = np.array(X_all, dtype=np.float32)
        y = encoder.transform(y_all)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))

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
        out = Dense(len(CLASSES), activation='softmax')(x)
        model = Model(inp, out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        save_path = os.path.join(save_dir, f'model_{floor}.keras')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        trained_any = True
        print(f'[TRAIN] 바닥={floor} | 파일수={used_files}/{len(file_list)} | 세그먼트수={used_segments} | 샘플수={len(X)}')
        model.fit(
            X, y_cat,
            validation_split=0.2,
            epochs=60,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

    if not trained_any:
        print('[중단] 실제로 학습에 들어간 바닥이 없습니다. 파일명 키워드, CSV 형식, 경로를 확인하세요.')


if __name__ == '__main__':
    train_models()
