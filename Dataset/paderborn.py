# ==============================================================================
# Modified by Dayoon (2026) for Multi-Load / Multi-Domain TSAD Experiments
# Description: Upgraded Paderborn Loader to support a list of operational settings.
# ==============================================================================

import os
import numpy as np
import scipy.io
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Paderborn 데이터 기본 위치: 이 파일(MTGFLOW/Dataset/)의 두 단계 상위가 MTGFLOW,
# 그 옆(../)의 Data/Paderborn. 실행 위치(cwd)와 무관하게 항상 올바른 경로를 가리킨다.
_DATA_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', 'Data', 'Paderborn'))

PADERBORN_SETTING_META = {
    "N15_M07_F10": [1500.0, 0.7, 1000.0],
    "N09_M07_F10": [900.0, 0.7, 1000.0],
    "N15_M01_F10": [1500.0, 0.1, 1000.0],
    "N15_M07_F04": [1500.0, 0.7, 400.0],
}


def normalize_paderborn_meta(meta):
    rpm, torque, force = meta
    return np.array([
        (rpm - 1500.0) / 1500.0,
        (torque - 0.7) / 0.7,
        (force - 1000.0) / 1000.0,
    ], dtype=np.float32)


def get_paderborn_setting_meta(setting_name):
    if setting_name not in PADERBORN_SETTING_META:
        raise ValueError(f"Unknown Paderborn setting {setting_name}. Add it to PADERBORN_SETTING_META before using metadata context.")
    return normalize_paderborn_meta(PADERBORN_SETTING_META[setting_name])

def _sensor_name_to_str(name):
    arr = np.asarray(name).squeeze()
    if arr.shape == ():
        return str(arr.item())
    return ''.join(str(item) for item in arr.tolist())


def extract_signal_from_mat(mat, key, sensor_name):
    y_array = mat[key][0, 0]["Y"]

    for i in range(y_array.shape[1]):
        sensor_room = y_array[0, i]
        name = _sensor_name_to_str(sensor_room["Name"][0])

        if name == sensor_name:
            return sensor_room["Data"].flatten()

    raise ValueError(f"Sensor {sensor_name} not found in {key}")


def _normalize_sensor_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _matches_measured_sensor(sensor_name, target):
    normalized = _normalize_sensor_name(sensor_name)
    if target == "speed":
        return "speed" in normalized or normalized in {"n", "rpm"}
    if target == "torque":
        return "torque" in normalized or normalized.startswith("m")
    if target == "force":
        return "force" in normalized and ("radial" in normalized or normalized.startswith("fr") or normalized == "force")
    return False


def load_measured_operational_signals(mat_path):
    mat = scipy.io.loadmat(mat_path)
    key = os.path.basename(mat_path).replace(".mat", "")
    y_array = mat[key][0, 0]["Y"]
    signals = {}

    for i in range(y_array.shape[1]):
        sensor_room = y_array[0, i]
        name = _sensor_name_to_str(sensor_room["Name"][0])
        data = sensor_room["Data"].flatten()
        for target in ("speed", "torque", "force"):
            if target not in signals and _matches_measured_sensor(name, target):
                signals[target] = data

    missing = [target for target in ("speed", "torque", "force") if target not in signals]
    if missing:
        available = [_sensor_name_to_str(y_array[0, i]["Name"][0]) for i in range(y_array.shape[1])]
        raise ValueError(f"Measured operational sensor(s) {missing} not found in {key}. Available sensors: {available}")

    min_len = min(len(signals[target]) for target in ("speed", "torque", "force"))
    return np.stack([signals[target][:min_len] for target in ("speed", "torque", "force")], axis=1).astype(np.float32)


def summarize_measured_meta(op_signals, start, end, measured_meta_stats, vibration_sampling_rate=64000, op_sampling_rate=4000):
    op_start = int(round(start * op_sampling_rate / vibration_sampling_rate))
    op_end = int(round(end * op_sampling_rate / vibration_sampling_rate))
    op_start = max(0, min(op_start, len(op_signals) - 1))
    op_end = max(op_start + 1, min(op_end, len(op_signals)))
    window = op_signals[op_start:op_end]
    means = window.mean(axis=0)
    if measured_meta_stats == "mean":
        return means.astype(np.float32)
    stds = window.std(axis=0)
    return np.array([means[0], stds[0], means[1], stds[1], means[2], stds[2]], dtype=np.float32)


class Paderborn_dataset(Dataset):
    def __init__(self, windows, labels, window_size, ids=None, metas=None, rms_z=None) -> None:
        super(Paderborn_dataset, self).__init__()
        self.windows = windows
        self.label = labels
        self.window_size = window_size
        self.ids = np.array(ids) if ids is not None else np.array(['unknown'] * len(windows))
        self.metas = np.asarray(metas, dtype=np.float32) if metas is not None else np.zeros((len(windows), 3), dtype=np.float32)
        # 작업 D(진폭 confound 교정): window별 z-scored log-RMS를 스코어 페널티용 feature로 carry.
        # 진폭 정규화(amp_normalize)를 쓰지 않는 실행에선 0으로 채워 하위호환.
        self.rms_z = (np.asarray(rms_z, dtype=np.float32).reshape(-1, 1)
                      if rms_z is not None else np.zeros((len(windows), 1), dtype=np.float32))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        # [Batch, Length, 1] -> MTGFlow 규격 맞춤 [1, Length, 1]
        window_data = self.windows[index].reshape([self.window_size, -1, 1])
        return (torch.FloatTensor(window_data).transpose(0, 1), self.label[index], index,
                torch.FloatTensor(self.metas[index]), torch.FloatTensor(self.rms_z[index]))


def loader_Paderborn_OCC(root=_DATA_ROOT,
                         loads=["N15_M07_F10"],
                         train_loads=None,
                         test_loads=None,
                         batch_size=64, 
                         window_size=2048, 
                         stride_size=1024, 
                         label=False,
                         train_ids=['K001', 'K002', 'K003'], 
                         val_ids=['K004'], 
                         test_norm_ids=['K005', 'K006'],
                         exclude_ids=None,
                         sensor_mode='vibration_1',
                         meta_source='static',
                         measured_meta_stats='meanstd',
                         amp_normalize=False,
                         rms_eps=1e-8,
                         vibration_sampling_rate=64000,
                         op_sampling_rate=4000):
    """
    여러 세팅 폴더를 동시에 읽어와 통합 학습/추론이 가능한 OCC 데이터 로더.

    기본값은 기존 pooled multi-setting 동작과 동일하게 ``loads``의 모든 setting에서
    train/val/test를 모두 수집한다. ``train_loads``와 ``test_loads``가 주어지면
    train/val은 train setting에서만, test normal/fault는 test setting에서만 수집하는
    cross-domain split으로 동작한다. StandardScaler는 항상 train setting의 train normal
    파일에만 fit되고, val/test에는 동일 scaler의 transform만 적용된다.
    """
    
    def normalize_id_list(ids):
        normalized = []
        for item in ids or []:
            normalized.extend(part.strip() for part in str(item).split(',') if part.strip())
        return normalized

    train_ids = set(normalize_id_list(train_ids))
    val_ids = set(normalize_id_list(val_ids))
    test_norm_ids = set(normalize_id_list(test_norm_ids))
    exclude_ids = set(normalize_id_list(exclude_ids))
    if meta_source not in ('static', 'measured'):
        raise ValueError("meta_source must be one of ['static', 'measured']")
    if measured_meta_stats not in ('mean', 'meanstd'):
        raise ValueError("measured_meta_stats must be one of ['mean', 'meanstd']")
    meta_input_dim = 3 if meta_source == 'static' or measured_meta_stats == 'mean' else 6
    loads = normalize_id_list(loads)
    if (train_loads is None) != (test_loads is None):
        raise ValueError("--train_load_setting and --test_load_setting must be provided together for cross-domain Paderborn loading.")
    if train_loads is None and test_loads is None:
        mode = 'pooled'
        train_loads = list(loads)
        test_loads = list(loads)
    else:
        mode = 'cross-domain'
        train_loads = normalize_id_list(train_loads)
        test_loads = normalize_id_list(test_loads)

    def extract_bearing_id(filename):
        match = re.search(r'(K[A-Z]?\d{2,3})(?:_|\.)', filename)
        return match.group(1) if match else filename.replace('.mat', '')

    # 여러 도메인의 파일들을 추적하기 위해 (폴더절대경로, 파일명) 튜플 형태로 수집 리스트 선언
    train_file_tuples = []
    val_file_tuples = []
    test_normal_file_tuples = []
    test_fault_file_tuples = []

    def iter_setting_files(setting_list):
        for load_setting in setting_list:
            setting_path = os.path.join(root, load_setting)
            if not os.path.exists(setting_path):
                print(f"⚠️ 경고: 해당 운전 조건 폴더가 경로에 없어 건너뜁니다: {setting_path}")
                continue

            filenames = [f for f in os.listdir(setting_path) if f.endswith('.mat')]
            for f in filenames:
                bearing_id = extract_bearing_id(f)
                if bearing_id in exclude_ids:
                    continue
                yield setting_path, f, bearing_id

    # 🚀 Step 1: train settings에서는 train/val만, test settings에서는 test만 수집
    for setting_path, f, bearing_id in iter_setting_files(train_loads):
        file_info = (setting_path, f)
        if bearing_id in train_ids:
            train_file_tuples.append(file_info)
        elif bearing_id in val_ids:
            val_file_tuples.append(file_info)

    for setting_path, f, bearing_id in iter_setting_files(test_loads):
        file_info = (setting_path, f)
        if bearing_id in test_norm_ids:
            test_normal_file_tuples.append(file_info)
        elif bearing_id not in train_ids and bearing_id not in val_ids:
            # 지정된 정상 계열 외의 모든 실제 결함 파일들 수집
            test_fault_file_tuples.append(file_info)

    if len(train_file_tuples) == 0:
        raise ValueError("❌ 지정된 조건에 맞는 학습 데이터 파일(.mat)을 찾을 수 없습니다.")

    # 🚀 Step 2: 다중 도메인 통합 StandardScaler 피팅 (Data Leakage 절대 방지)
    # 여러 하중 환경의 정상 신호를 모두 모아서 단 하나의 글로벌 스케일러를 학습시킵니다.
    raw_train_signals = []
    for folder_path, f in train_file_tuples:
        mat = scipy.io.loadmat(os.path.join(folder_path, f))
        key = f.replace('.mat', '')
        for j in mat[key][0][0]:
            if 'Name' in str(j.dtype) and sensor_mode in j[0]['Name']:
                idx = np.argwhere(j[0]['Name'] == sensor_mode)
                raw_train_signals.append(j[0]['Data'][idx][0][0][0])
                break
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate(raw_train_signals).reshape(-1, 1))

    # 🚀 Step 3: 파일 경계면 브레이크 없이 윈도우를 추출하는 내부 헬퍼 함수
    def extract_scaled_windows(file_tuple_list):
        extracted_windows = []
        extracted_ids = []
        extracted_metas = []
        extracted_rms = []  # 작업 D: window별 원 RMS(정규화 전, 비-z-score)
        setting_window_counts = {}
        for folder_path, f in file_tuple_list:
            bearing_id = extract_bearing_id(f)
            setting_name = os.path.basename(folder_path)
            static_setting_meta = get_paderborn_setting_meta(setting_name)
            mat_path = os.path.join(folder_path, f)
            measured_signals = load_measured_operational_signals(mat_path) if meta_source == 'measured' else None
            mat = scipy.io.loadmat(mat_path)
            key = f.replace('.mat', '')
            sig = None
            for j in mat[key][0][0]:
                if 'Name' in str(j.dtype) and sensor_mode in j[0]['Name']:
                    idx = np.argwhere(j[0]['Name'] == sensor_mode)
                    sig = j[0]['Data'][idx][0][0][0]
                    break
            if sig is None:
                continue

            # 통합 스케일러로 정규화 매핑 후 1차원 플래튼
            scaled_sig = scaler.transform(sig.reshape(-1, 1)).flatten()

            # 파일 단위 경계면을 침범하지 않는 독립 슬라이딩
            start = 0
            while start + window_size <= len(scaled_sig):
                w = scaled_sig[start:start + window_size]
                # 작업 D 진폭 교정: per-window RMS(평균 제거 X, 작업 B 정의와 동일 sqrt(mean(x²)))를
                # 떼어 shape에 집중. amp_normalize면 window를 RMS로 나눠 단위진폭(mean(x²)≈1)으로 만들고,
                # 떼어낸 원 RMS는 별도 보존(게이트 (a)용 원 RMS + 스코어 페널티용 z-score 재료).
                rms = float(np.sqrt(np.mean(w ** 2)))
                if amp_normalize:
                    w = w / (rms + rms_eps)
                extracted_windows.append(w)
                extracted_rms.append(rms)
                extracted_ids.append(bearing_id)
                if meta_source == 'measured':
                    meta = summarize_measured_meta(
                        measured_signals, start, start + window_size, measured_meta_stats,
                        vibration_sampling_rate=vibration_sampling_rate,
                        op_sampling_rate=op_sampling_rate,
                    )
                else:
                    meta = static_setting_meta
                extracted_metas.append(meta)
                setting_window_counts[setting_name] = setting_window_counts.get(setting_name, 0) + 1
                start += stride_size
                
        if not extracted_windows:
            return (np.empty((0, window_size)), np.array([], dtype=str),
                    np.empty((0, meta_input_dim), dtype=np.float32),
                    np.empty((0,), dtype=np.float32), setting_window_counts)
        return (np.array(extracted_windows), np.array(extracted_ids),
                np.array(extracted_metas, dtype=np.float32),
                np.array(extracted_rms, dtype=np.float32), setting_window_counts)

    # 🚀 Step 4: 멀티 도메인 데이터셋 윈도우 가공 및 빌딩
    train_x, train_ids_per_window, train_meta, train_rms, train_setting_counts = extract_scaled_windows(train_file_tuples)
    val_x, val_ids_per_window, val_meta, val_rms, val_setting_counts = extract_scaled_windows(val_file_tuples)
    test_norm_x, test_norm_ids_per_window, test_norm_meta, test_norm_rms, test_norm_setting_counts = extract_scaled_windows(test_normal_file_tuples)
    test_fault_x, test_fault_ids_per_window, test_fault_meta, test_fault_rms, test_fault_setting_counts = extract_scaled_windows(test_fault_file_tuples)

    test_x = np.concatenate([test_norm_x, test_fault_x], axis=0)
    test_ids_per_window = np.concatenate([test_norm_ids_per_window, test_fault_ids_per_window], axis=0)
    test_meta = np.concatenate([test_norm_meta, test_fault_meta], axis=0)
    test_rms = np.concatenate([test_norm_rms, test_fault_rms], axis=0)

    # 작업 D: log-RMS를 train-normal 기준 z-score → 스코어 페널티(z_rms)용 feature.
    # measured meta z-score(누수 방지, train stats로만 fit)와 동일 규약. amp_normalize=False여도
    # 통계는 계산해 두되 rms_z가 스코어에 쓰이는지는 test.py의 rms_lambda가 결정한다.
    train_logrms = np.log(train_rms + rms_eps) if len(train_rms) else train_rms
    val_logrms = np.log(val_rms + rms_eps) if len(val_rms) else val_rms
    test_logrms = np.log(test_rms + rms_eps) if len(test_rms) else test_rms
    if len(train_logrms):
        train_logrms_mean = float(train_logrms.mean())
        train_logrms_std = float(train_logrms.std())
    else:
        train_logrms_mean, train_logrms_std = 0.0, 1.0

    def _zscore_logrms(a):
        if not len(a):
            return a.astype(np.float32) if hasattr(a, 'astype') else np.empty((0,), dtype=np.float32)
        return ((a - train_logrms_mean) / (train_logrms_std + 1e-8)).astype(np.float32)

    train_rms_z = _zscore_logrms(train_logrms)
    val_rms_z = _zscore_logrms(val_logrms)
    test_rms_z = _zscore_logrms(test_logrms)

    if meta_source == 'measured':
        train_meta_mean = train_meta.mean(axis=0)
        train_meta_std = train_meta.std(axis=0)
        train_meta = (train_meta - train_meta_mean) / (train_meta_std + 1e-8)
        val_meta = (val_meta - train_meta_mean) / (train_meta_std + 1e-8) if len(val_meta) else val_meta
        test_meta = (test_meta - train_meta_mean) / (train_meta_std + 1e-8) if len(test_meta) else test_meta
    
    # 이진 라벨 정의 (정상 0, 이상 1)
    train_y = np.zeros(len(train_x))
    val_y = np.zeros(len(val_x))
    test_y = np.array([0] * len(test_norm_x) + [1] * len(test_fault_x))

    n_sensor = 1 

    print(f'Mode: {mode}')
    print(f'Train Settings: {train_loads}')
    print(f'Test Settings: {test_loads}')
    print(f'Sensor Mode: {sensor_mode}')
    print(f'Sensor Names: {[sensor_mode]}')
    print(f'Metadata Source: {meta_source}')
    print(f'Measured Metadata Stats: {measured_meta_stats}')
    print(f'Metadata Input Dim: {meta_input_dim}')
    print(f'n_sensor: {n_sensor}')
    if exclude_ids:
        print(f'Excluded Bearing IDs: {sorted(exclude_ids)}')
    print(f'Total Train Windows: {len(train_x)}')
    print(f'Val Windows: {len(val_x)}')
    print(f'Test Windows: {len(test_x)}')

    def print_setting_summary(split_name, counts):
        print(f'{split_name} Setting Window Counts:')
        for setting_name in sorted(counts):
            meta = get_paderborn_setting_meta(setting_name).tolist()
            raw_meta = PADERBORN_SETTING_META[setting_name]
            print(f'  {setting_name}: windows={counts[setting_name]}, raw_meta={raw_meta}, normalized_meta={meta}')

    print_setting_summary('Train', train_setting_counts)
    print_setting_summary('Val', val_setting_counts)
    merged_test_counts = {}
    for counts in (test_norm_setting_counts, test_fault_setting_counts):
        for setting_name, count in counts.items():
            merged_test_counts[setting_name] = merged_test_counts.get(setting_name, 0) + count
    print_setting_summary('Test', merged_test_counts)

    if amp_normalize:
        print(f'Amplitude Normalize: per-window RMS (log-RMS train z-score: mean={train_logrms_mean:.4f}, std={train_logrms_std:.4f})')

    # 파이토치 데이터로더 패킹 및 반환
    train_ds = Paderborn_dataset(train_x, train_y, window_size, train_ids_per_window, train_meta, train_rms_z)
    val_ds = Paderborn_dataset(val_x, val_y, window_size, val_ids_per_window, val_meta, val_rms_z)
    test_ds = Paderborn_dataset(test_x, test_y, window_size, test_ids_per_window, test_meta, test_rms_z)

    # 작업 D: 게이트 (a)용 원 RMS(정규화 전, 비-z-score)와 train z-score 통계를 dataset에 노출.
    # 진단 스크립트가 test_loader.dataset.rms_raw / train_logrms_mean 등으로 접근한다.
    for ds, rms_arr in ((train_ds, train_rms), (val_ds, val_rms), (test_ds, test_rms)):
        ds.rms_raw = np.asarray(rms_arr, dtype=np.float32)
        ds.train_logrms_mean = train_logrms_mean
        ds.train_logrms_std = train_logrms_std
        ds.amp_normalize = bool(amp_normalize)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=not label)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor


if __name__ == '__main__':
    # 로컬 가동 및 차원 디버깅 테스트용 플래그
    # 세팅 0(N15_M07_F10)과 세팅 2(N15_M01_F10) 데이터를 동시에 묶어서 로드하는 예시
    train_l, val_l, test_l, ns = loader_Paderborn_OCC(
        file_path=_DATA_ROOT,
        loads=['N15_M07_F10', 'N15_M01_F10'], # 🚀 두 개 폴더 동시 지정 테스트
        window_size=2048,
        stride_size=1024
    )