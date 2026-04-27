import csv
import glob
import os
import pickle
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import LSTM, BatchNormalization, Bidirectional, Conv1D, Dense, Dropout, Input, MaxPooling1D
from tensorflow.keras.models import Model


DATA_DIR = r"C:\Users\MASL\Desktop\코드개선\3차_4분할\trimmed_manual_split_4phase"
SAVE_DIR = "./kumoh_lstm_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 15
SMOOTH_K = 5
RANDOM_SEED = 42
AUG_PER_WINDOW = 2
MIN_STEP_WINDOWS = 25
MIN_SEGMENT_POINTS = 4

# 4분할 구간별 보존 기준
PHASE_ACTIVE_THRESHOLDS = {
    'rise': 4.5,
    'plateau': 5.0,
    'fall': 4.5,
    'floor_return': 2.5,
}
PHASE_ACTIVE_RATIOS = {
    'rise': 0.18,
    'plateau': 0.35,
    'fall': 0.18,
    'floor_return': 0.00,
}
PHASE_AUG_MULTIPLIER = {
    'rise': 3,
    'plateau': 2,
    'fall': 3,
    'floor_return': 1,
}

ALL_CLASSES = ['바닥', '물', '말차', '커피', '콜라', '토마토', '우유', '망고', '기름', '수박']
LIQUID_ALIASES = {
    'water': '물', 'coffee': '커피', 'cola': '콜라', 'milk': '우유',
    'mango': '망고', 'oil': '기름', 'matcha': '말차', 'tomato': '토마토', 'watermelon': '수박',
    '물': '물', '커피': '커피', '콜라': '콜라', '우유': '우유', '망고': '망고', '기름': '기름',
    '말차': '말차', '토마토': '토마토', '수박': '수박'
}
FLOOR_KEYWORDS = {
    '검대': ['검대'],
    '회대': ['회대'],
    '황대': ['황대'],
    '207회바': ['greyfloor', '회색바닥', '207회바'],
    '흰책상': ['white', '하양', '흰', 'whitedesk', '흰책상'],
    '나타': ['나타'],
    '회타': ['회타'],
    '나무': ['나무', 'wood'],
    '그마': ['그마', 'greymarble'],
}
REVERSE_DIRECTION_FLOORS = {'검대'}

PHASE_FOLDERS = ['rise', 'plateau', 'fall', 'floor_return']
LIQUID_PHASES = {'rise', 'plateau', 'fall'}
FLOOR_PHASES = {'floor_return'}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def decode_hash_u(text: str) -> str:
    def repl(match):
        code = match.group(1)
        try:
            return chr(int(code, 16))
        except Exception:
            return match.group(0)
    return re.sub(r'#U([0-9A-Fa-f]{4,6})', repl, text)


def normalized_text(text: str) -> str:
    return decode_hash_u(str(text)).lower()


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


def detect_phase_type(fp: str):
    name = normalized_text(os.path.basename(fp))
    parent = normalized_text(os.path.basename(os.path.dirname(fp)))
    full = normalized_text(fp)
    for phase in PHASE_FOLDERS:
        key = phase.lower()
        if f'_{key}_' in name or parent == key or f'/{key}/' in full.replace('\\', '/'):
            return phase
    return None


def is_liquid_phase_file(fp: str) -> bool:
    phase = detect_phase_type(fp)
    return phase in LIQUID_PHASES


def is_floor_phase_file(fp: str) -> bool:
    phase = detect_phase_type(fp)
    return phase in FLOOR_PHASES


def detect_floor_type(text: str):
    lower = normalized_text(text)
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None


def detect_floor_type_from_path(fp: str):
    parts = [os.path.basename(fp), os.path.dirname(fp), str(Path(fp).parent)]
    for part in parts:
        floor = detect_floor_type(part)
        if floor:
            return floor
    return None


def detect_liquid_from_filename(filename: str):
    lower = normalized_text(filename)
    for key, kor in LIQUID_ALIASES.items():
        if key.lower() in lower:
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
        if (
            abs(float(sig[i]) - local_med) > k * local_std
            and abs(float(sig[i]) - float(sig[i - 1])) > 1.2 * local_std
            and abs(float(sig[i]) - float(sig[i + 1])) > 1.2 * local_std
        ):
            sig[i] = np.float32(0.5 * (sig[i - 1] + sig[i + 1]))
    return sig


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

        if len(currents) < MIN_SEGMENT_POINTS:
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

        if len(currents) < MIN_SEGMENT_POINTS:
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
        return 0.020, 0.65, 5.0
    if floor_type == '황대':
        return 0.028, 0.75, 6.0
    return 0.026, 0.75, 6.0


def build_recent_floor_reference(signal, baseline, floor_type):
    alpha, ref_grad, ref_band = get_floor_ref_params(floor_type)
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


def pad_segment_to_window(arr, target_len=WINDOW_SIZE, mode='edge'):
    arr = np.asarray(arr, dtype=np.float32)
    n = len(arr)
    if n >= target_len:
        return arr[:target_len].copy()
    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    if n == 0:
        return np.zeros((target_len,), dtype=np.float32)
    if mode == 'reflect' and n >= 2:
        return np.pad(arr, (pad_left, pad_right), mode='reflect').astype(np.float32)
    return np.pad(arr, (pad_left, pad_right), mode='edge').astype(np.float32)


def make_feature_window(signal, start, base, floor_type, segment_end=None):
    if segment_end is None:
        raw_sig = signal[start:start + WINDOW_SIZE]
    else:
        raw_sig = signal[start:segment_end]

    sig = pad_segment_to_window(raw_sig, WINDOW_SIZE)
    norm = (sig - sig.mean()) / (sig.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])

    base_eff = base if len(signal) < 20 else 0.7 * base + 0.3 * np.mean(signal[:20])
    ctx_start = max(0, start - 5)
    if segment_end is None:
        ctx_end = min(len(signal), start + WINDOW_SIZE + 5)
    else:
        ctx_end = min(len(signal), segment_end + 5)
    ctx_mean = np.mean(signal[ctx_start:ctx_end])
    direction_raw = base_eff - ctx_mean if floor_type in REVERSE_DIRECTION_FLOORS else ctx_mean - base_eff
    direction = np.clip(direction_raw / 300.0, -1.0, 1.0)
    direction_ch = np.full(WINDOW_SIZE, direction, dtype=np.float32)
    return np.stack([norm, d1, d2, direction_ch]).astype(np.float32)


def should_keep_phase_window(chg_window: np.ndarray, phase: str) -> bool:
    thr = PHASE_ACTIVE_THRESHOLDS.get(phase, 5.0)
    ratio_thr = PHASE_ACTIVE_RATIOS.get(phase, 0.2)
    active_ratio = float(np.mean(chg_window >= thr))
    mid = chg_window[WINDOW_SIZE // 3: WINDOW_SIZE - WINDOW_SIZE // 3]
    center_peak = float(np.max(mid)) if len(mid) else float(np.max(chg_window))
    full_peak = float(np.max(chg_window))

    if phase == 'plateau':
        return active_ratio >= ratio_thr or center_peak >= thr
    if phase in {'rise', 'fall'}:
        grad = np.gradient(chg_window)
        dynamic = float(np.max(np.abs(grad)))
        return active_ratio >= ratio_thr or full_peak >= thr or dynamic >= 0.8
    if phase == 'floor_return':
        return True
    return active_ratio >= ratio_thr or full_peak >= thr


def build_dataset_for_floor(file_list, floor_type):
    X_all, y_all, meta = [], [], []
    present_classes = {'바닥'}
    phase_counter = {p: 0 for p in PHASE_FOLDERS}

    for fp in sorted(file_list):
        phase = detect_phase_type(fp)
        if phase is None:
            print(f'[SKIP] 구간 타입 인식 실패: {os.path.basename(fp)}')
            continue

        base, signal, changes = read_csv_recalc_change(fp)
        if base is None:
            print(f'[SKIP] 읽기 실패: {os.path.basename(fp)}')
            continue

        smooth_sig, _ = build_recent_floor_reference(signal, base, floor_type)
        signal_len = len(signal)
        max_start = signal_len - WINDOW_SIZE

        segment_type = 'floor' if phase in FLOOR_PHASES else 'liquid'
        liquid_label = detect_liquid_from_filename(os.path.basename(fp)) if segment_type == 'liquid' else '바닥'
        if segment_type == 'liquid' and liquid_label is None:
            print(f'[SKIP] 액체명 인식 실패: {os.path.basename(fp)}')
            continue
        if liquid_label:
            present_classes.add(liquid_label)

        aug_count = max(0, AUG_PER_WINDOW * PHASE_AUG_MULTIPLIER.get(phase, 1) - 1)

        if signal_len < WINDOW_SIZE:
            starts = [0]
            short_segment_mode = True
            step = 1
        else:
            step = max(1, max_start // MIN_STEP_WINDOWS)
            starts = list(range(0, max_start + 1, step))
            short_segment_mode = False

        for start in starts:
            segment_end = signal_len if short_segment_mode else start + WINDOW_SIZE
            raw_chg_w = changes[start:segment_end]
            if len(raw_chg_w) < MIN_SEGMENT_POINTS:
                continue

            if short_segment_mode:
                chg_w = pad_segment_to_window(raw_chg_w, WINDOW_SIZE)
                keep_window = True if phase in {'rise', 'fall'} else should_keep_phase_window(chg_w, phase)
            else:
                chg_w = raw_chg_w
                keep_window = should_keep_phase_window(chg_w, phase)

            if not keep_window:
                continue

            label = '바닥' if segment_type == 'floor' else liquid_label
            win = make_feature_window(smooth_sig, start, base, floor_type, segment_end=segment_end if short_segment_mode else None)

            for aug_idx in range(aug_count + 1):
                if aug_idx == 0:
                    aug = win.copy()
                else:
                    aug = win + np.random.normal(0, random.uniform(0.01, 0.04), win.shape)
                    aug[:3] *= random.uniform(0.95, 1.05)
                X_all.append(aug.astype(np.float32))
                y_all.append(label)
                meta.append({
                    'floor': floor_type,
                    'source_file': decode_hash_u(os.path.basename(fp)),
                    'phase': phase,
                    'segment_type': segment_type,
                    'start_idx': start,
                    'end_idx': segment_end - 1,
                    'raw_length': int(len(raw_chg_w)),
                    'padded_to_window': 1 if short_segment_mode else 0,
                    'label': label,
                    'max_change': float(np.max(raw_chg_w)),
                    'mean_change': float(np.mean(raw_chg_w)),
                    'active_ratio': float(np.mean(raw_chg_w >= PHASE_ACTIVE_THRESHOLDS.get(phase, 5.0))),
                    'augmented': 0 if aug_idx == 0 else 1,
                })
                phase_counter[phase] += 1

    return X_all, y_all, meta, sorted(present_classes), phase_counter


def save_dataset_manifest(meta, save_dir, floor_type):
    path = os.path.join(save_dir, f'{floor_type}_train_manifest_4phase.csv')
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        writer.writeheader()
        writer.writerows(meta)
    return path


def save_confusion_matrix(y_true, y_pred, labels, save_dir, floor_type):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
    plt.tight_layout()
    out = os.path.join(save_dir, f'{floor_type}_confusion_matrix_4phase.png')
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out


def train_models(data_dir=DATA_DIR, save_dir=SAVE_DIR):
    floor_files = {f: [] for f in FLOOR_KEYWORDS}
    for fp in iter_csv_files(data_dir):
        phase = detect_phase_type(fp)
        if phase is None:
            continue
        floor = detect_floor_type_from_path(fp)
        if floor:
            floor_files[floor].append(fp)

    for floor, file_list in floor_files.items():
        if not file_list:
            continue

        X_all, y_all, meta, present_classes, phase_counter = build_dataset_for_floor(file_list, floor)
        if len(X_all) == 0:
            print(f'[SKIP] {floor}: 학습 데이터가 없습니다.')
            continue

        encoder = LabelEncoder().fit(present_classes)
        with open(os.path.join(save_dir, f'encoder_{floor}.pkl'), 'wb') as f:
            pickle.dump(encoder, f)
        with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as f:
            pickle.dump(encoder, f)

        X = np.array(X_all, dtype=np.float32).transpose(0, 2, 1)
        y = encoder.transform(y_all)
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(encoder.classes_))

        manifest_path = save_dataset_manifest(meta, save_dir, floor)
        print(f'[INFO] {floor} manifest 저장: {manifest_path}')
        print(f'[INFO] {floor} 사용 클래스: {list(encoder.classes_)}')
        print(f'[INFO] {floor} phase 샘플 수: {phase_counter}')

        unique, counts = np.unique(y, return_counts=True)
        print('[INFO] 클래스별 샘플 수:', {encoder.classes_[i]: int(c) for i, c in zip(unique, counts)})

        stratify_y = y if np.min(counts) >= 2 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_cat, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify_y, shuffle=True
        )

        y_train_int = np.argmax(y_train, axis=1)
        classes_present = np.unique(y_train_int)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y_train_int)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, class_weights)}

        inp = Input(shape=(WINDOW_SIZE, 4))
        x = Conv1D(32, 3, padding='same', activation='relu')(inp)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64, return_sequences=False))(x)
        x = Dropout(0.3)(x)
        out = Dense(len(encoder.classes_), activation='softmax')(x)
        model = Model(inp, out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        save_path = os.path.join(save_dir, f'model_{floor}.keras')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ]

        print(f'[TRAIN-4PHASE] 바닥={floor} | 파일수={len(file_list)} | 샘플수={len(X)}')
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )

        val_pred = model.predict(X_val, verbose=0)
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(val_pred, axis=1)

        cm_path = save_confusion_matrix(y_true, y_pred, list(encoder.classes_), save_dir, floor)
        report_path = os.path.join(save_dir, f'{floor}_classification_report_4phase.txt')
        with open(report_path, 'w', encoding='utf-8-sig') as f:
            f.write(classification_report(y_true, y_pred, target_names=list(encoder.classes_), digits=4, zero_division=0))

        hist_path = os.path.join(save_dir, f'{floor}_history_4phase.csv')
        with open(hist_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss'])
            for i in range(len(history.history['loss'])):
                writer.writerow([
                    i + 1,
                    history.history['accuracy'][i],
                    history.history['loss'][i],
                    history.history['val_accuracy'][i],
                    history.history['val_loss'][i],
                ])

        print(f'[SAVE] 모델: {save_path}')
        print(f'[SAVE] 컨퓨전매트릭스: {cm_path}')
        print(f'[SAVE] 리포트: {report_path}')
        print(f'[SAVE] 히스토리: {hist_path}')


if __name__ == '__main__':
    train_models()
