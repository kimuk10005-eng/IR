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

DATA_DIR = r"C:\Users\MASL\Desktop\ir data(get)\260322"
SAVE_DIR = "./kumoh_lstm_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_SIZE = 15
CHANGE_THRESHOLD = 5.0
LOCAL_CHANGE_THRESHOLD = 4.5
LOCAL_MEAN_THRESHOLD = 4.0
LOCAL_MAX_FOR_MEAN_THRESHOLD = 8.0
LOCAL_AREA_THRESHOLD = 90.0
LOCAL_MAX_FOR_AREA_THRESHOLD = 10.0
LOCAL_REF_BACK = 20
LOCAL_REF_GUARD = 3
SMOOTH_KERNEL = 5

LIQUIDS = ['바닥', '물', '말차', '커피', '콜라', '토마토', '우유', '망고', '기름', '수박']

LIQUID_ALIASES = {
    'water': '물', 'coffee': '커피', 'cola': '콜라', 'milk': '우유',
    'mango': '망고', 'oil': '기름', 'matcha': '말차', 'tomato': '토마토', 'watermelon': '수박'
}

FLOOR_KEYWORDS = {
    '검대': ['검대'],
    '회대': ['회대'],
    '황대': ['황대'],
    '207회바': ['greyfloor', '회색바닥'],
    '흰책상': ['white', '하양', '흰', 'whitedesk'],
    '나타': ['나타'],
    '회타': ['회타'],
}

REVERSE_DIRECTION_FLOORS = {'검대'}


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


def read_csv_recalc_change(fp):
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    for enc in encodings:
        try:
            baseline_values = []
            currents = []
            row_bases = []

            with open(fp, 'r', encoding=enc, errors='ignore', newline='') as f:
                reader = csv.DictReader(f)
                fields = [name.strip() for name in (reader.fieldnames or [])]
                if 'record_type' not in fields:
                    continue

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
                continue

            if baseline_values:
                base = float(np.mean(baseline_values))
            else:
                valid_row_bases = [b for b in row_bases if b is not None]
                if not valid_row_bases:
                    continue
                base = float(np.mean(valid_row_bases))

            currents = np.array(currents, dtype=np.float32)
            changes = np.abs(currents - base).astype(np.float32)
            return base, currents, changes

        except Exception:
            continue

    return None, None, None


def create_window_with_label(signal, changes, start, base, floor_type, liquid):
    if start + WINDOW_SIZE > len(signal):
        return None, None

    sig = signal[start:start + WINDOW_SIZE]
    sig_smooth = smooth_signal(sig, kernel=min(SMOOTH_KERNEL, len(sig)))
    global_chg = changes[start:start + WINDOW_SIZE]

    norm = (sig_smooth - sig_smooth.mean()) / (sig_smooth.std() + 1e-8)
    d1 = np.diff(norm, prepend=norm[0])
    d1 = np.convolve(d1, np.ones(3, dtype=np.float32) / 3, mode='same')
    d2 = np.diff(d1, prepend=d1[0])

    local_ref = compute_local_reference(signal, start, base)
    if floor_type in REVERSE_DIRECTION_FLOORS:
        dynamic_ch = np.clip((local_ref - sig_smooth) / 40.0, -1.0, 1.0)
    else:
        dynamic_ch = np.clip((sig_smooth - local_ref) / 40.0, -1.0, 1.0)

    window = np.stack([norm, d1, d2, dynamic_ch], axis=0)

    local_diff = np.abs(sig - local_ref)
    local_change = float(np.max(local_diff))
    mean_local = float(np.mean(local_diff))
    area_local = float(np.sum(local_diff))
    max_global = float(np.max(global_chg))

    # 과검출을 줄이기 위해 area 단독 조건을 제거하고
    # local max / local mean / global change가 함께 있는 경우만 액체로 라벨링
    liquid_like = (
        (max_global >= CHANGE_THRESHOLD)
        or (local_change >= LOCAL_CHANGE_THRESHOLD and mean_local >= 2.8)
        or (mean_local >= LOCAL_MEAN_THRESHOLD and local_change >= LOCAL_MAX_FOR_MEAN_THRESHOLD)
        or (area_local >= LOCAL_AREA_THRESHOLD and local_change >= LOCAL_MAX_FOR_AREA_THRESHOLD)
    )

    label = liquid if liquid_like else '바닥'
    return window, label


def detect_floor_type(filename):
    lower = filename.lower()
    for floor, keywords in FLOOR_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return floor
    return None


def train_models(data_dir=DATA_DIR, save_dir=SAVE_DIR):
    global encoder
    encoder = LabelEncoder().fit(LIQUIDS)

    floor_files = {f: [] for f in FLOOR_KEYWORDS}
    for fp in glob.glob(os.path.join(data_dir, '*.csv')):
        name = os.path.basename(fp).lower()
        if any(skip in name for skip in ['바닥', 'base']):
            continue

        floor = detect_floor_type(name)
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

            max_start = len(signal) - WINDOW_SIZE
            if max_start <= 0:
                continue

            step = max(1, max_start // 25)

            for start in range(0, max_start + 1, step):
                win, label = create_window_with_label(signal, changes, start, base, floor, liquid)
                if win is None:
                    continue

                for _ in range(2):
                    aug = win + np.random.normal(0, random.uniform(0.01, 0.05), win.shape)
                    aug[:3] *= random.uniform(0.9, 1.1)
                    X_all.append(aug)
                    y_all.append(label)

        if len(X_all) == 0:
            print(f'[SKIP] {floor}: 학습 데이터가 없습니다. (파일명/파싱/임계값 확인)')
            continue

        X = np.array(X_all).transpose(0, 2, 1)
        y = encoder.transform(y_all)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(enumerate(class_weights))

        inputs = Input(shape=(WINDOW_SIZE, 4))
        x = Conv1D(32, 3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64, return_sequences=False))(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(len(encoder.classes_), activation='softmax')(x)
        model = Model(inputs, outputs)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model_path = os.path.join(save_dir, f'model_{floor}.keras')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
        ]

        print(f'[TRAIN] 바닥={floor} | 파일수={len(file_list)} | 샘플수={len(X_all)}')
        model.fit(
            X, y,
            validation_split=0.2,
            epochs=60,
            batch_size=32,
            verbose=1,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )

    with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)
    print('학습 완료 / encoder 저장 완료')


if __name__ == '__main__':
    train_models()
