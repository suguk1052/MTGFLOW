# ==============================================================================
# Modified by Dayoon (2026) for Multi-Load / Multi-Domain TSAD Experiments
# Description: Upgraded Paderborn Loader to support bandpass vibration input.
# ==============================================================================

import os
import re

import numpy as np
import scipy.io
from scipy.signal import butter, sosfiltfilt
import torch
from torch.utils.data import Dataset, DataLoader


class Paderborn_dataset(Dataset):
    def __init__(self, windows, labels, window_size, ids=None) -> None:
        super(Paderborn_dataset, self).__init__()
        self.windows = windows
        self.label = labels
        self.window_size = window_size
        self.ids = np.array(ids) if ids is not None else np.array(['unknown'] * len(windows))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        if window.ndim == 2:
            # Bandpass vibration representation: [n_bands, window_size] -> [n_bands, window_size, 1]
            window_data = window[:, :, np.newaxis]
        else:
            # Legacy 1-D representation: [window_size] -> [1, window_size, 1]
            window_data = window.reshape([self.window_size, -1, 1]).transpose(1, 0, 2)
        return torch.FloatTensor(window_data), self.label[index], index


def loader_Paderborn_OCC(root="/home/dayoon/DCP/Data/Paderborn",
                         loads=["N15_M07_F10"],
                         batch_size=64,
                         window_size=2048,
                         stride_size=1024,
                         label=False,
                         train_ids=['K001', 'K002', 'K003'],
                         val_ids=['K004'],
                         test_norm_ids=['K005', 'K006'],
                         exclude_ids=None,
                         sampling_rate=64000,
                         n_bands=8,
                         low_freq_start=1.0,
                         filter_order=4):
    """
    여러 세팅 폴더를 동시에 읽어와 통합 학습/추론이 가능한 OCC 데이터 로더.

    Paderborn 입력은 vibration_1만 사용하며, 각 파일의 전체 raw signal을 먼저
    bandpass filter bank로 [n_bands, time] representation으로 변환한 뒤 파일 경계를
    넘지 않도록 sliding window를 추출한다. Scaling은 train split의 정상 파일에서
    추출한 bandpass windows만으로 band별 mean/std를 fit하고 val/test/fault에는
    train에서 계산한 통계만 적용한다.
    """
    exclude_ids = _normalize_id_list(exclude_ids)
    train_ids = _drop_excluded_ids(_normalize_id_list(train_ids), exclude_ids)
    val_ids = _drop_excluded_ids(_normalize_id_list(val_ids), exclude_ids)
    test_norm_ids = _drop_excluded_ids(_normalize_id_list(test_norm_ids), exclude_ids)

    bands = _make_equal_band_ranges(
        sampling_rate=sampling_rate,
        n_bands=n_bands,
        low_freq_start=low_freq_start,
    )
    sos_bank = [_design_bandpass_sos(low, high, sampling_rate, filter_order) for low, high in bands]

    train_file_tuples = []
    val_file_tuples = []
    test_normal_file_tuples = []
    test_fault_file_tuples = []

    for load_setting in loads:
        setting_path = os.path.join(root, load_setting)
        if not os.path.exists(setting_path):
            print(f"⚠️ 경고: 해당 운전 조건 폴더가 경로에 없어 건너뜁니다: {setting_path}")
            continue

        filenames = sorted(f for f in os.listdir(setting_path) if f.endswith('.mat'))

        for filename in filenames:
            bearing_id = extract_bearing_id(filename)
            if bearing_id in exclude_ids:
                continue

            file_info = (setting_path, filename)
            if bearing_id in train_ids:
                train_file_tuples.append(file_info)
            elif bearing_id in val_ids:
                val_file_tuples.append(file_info)
            elif bearing_id in test_norm_ids:
                test_normal_file_tuples.append(file_info)
            else:
                test_fault_file_tuples.append(file_info)

    if len(train_file_tuples) == 0:
        raise ValueError("❌ 지정된 조건에 맞는 학습 데이터 파일(.mat)을 찾을 수 없습니다.")

    band_mean, band_std = _fit_band_scaler(
        train_file_tuples,
        sos_bank=sos_bank,
        window_size=window_size,
        stride_size=stride_size,
    )

    train_x, train_ids_per_window = _extract_bandpass_windows(
        train_file_tuples, sos_bank, window_size, stride_size, band_mean, band_std, n_bands
    )
    val_x, val_ids_per_window = _extract_bandpass_windows(
        val_file_tuples, sos_bank, window_size, stride_size, band_mean, band_std, n_bands
    )
    test_norm_x, test_norm_ids_per_window = _extract_bandpass_windows(
        test_normal_file_tuples, sos_bank, window_size, stride_size, band_mean, band_std, n_bands
    )
    test_fault_x, test_fault_ids_per_window = _extract_bandpass_windows(
        test_fault_file_tuples, sos_bank, window_size, stride_size, band_mean, band_std, n_bands
    )

    test_x = _concat_or_empty([test_norm_x, test_fault_x], n_bands, window_size)
    test_ids_per_window = np.concatenate([test_norm_ids_per_window, test_fault_ids_per_window], axis=0)

    train_y = np.zeros(len(train_x), dtype=np.int64)
    val_y = np.zeros(len(val_x), dtype=np.int64)
    test_y = np.array([0] * len(test_norm_x) + [1] * len(test_fault_x), dtype=np.int64)

    n_sensor = n_bands

    _print_paderborn_config(
        loads=loads,
        sampling_rate=sampling_rate,
        window_size=window_size,
        stride_size=stride_size,
        n_bands=n_bands,
        bands=bands,
        n_sensor=n_sensor,
        train_count=len(train_x),
        val_count=len(val_x),
        test_count=len(test_x),
        train_ids=train_ids,
        val_ids=val_ids,
        test_norm_ids=test_norm_ids,
        exclude_ids=exclude_ids,
    )

    train_loader = DataLoader(Paderborn_dataset(train_x, train_y, window_size, train_ids_per_window), batch_size=batch_size, shuffle=not label)
    val_loader = DataLoader(Paderborn_dataset(val_x, val_y, window_size, val_ids_per_window), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Paderborn_dataset(test_x, test_y, window_size, test_ids_per_window), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor


def extract_bearing_id(filename):
    match = re.search(r'(K[A-Z]?\d{2,3})(?:_|\.)', filename)
    return match.group(1) if match else filename.replace('.mat', '')


def _normalize_id_list(ids):
    if ids is None:
        return []
    if isinstance(ids, str):
        ids = [ids]

    normalized_ids = []
    for id_value in ids:
        for token in str(id_value).split(','):
            normalized_id = token.strip().strip('[](){}\"\'')
            if normalized_id:
                normalized_ids.append(normalized_id)
    return normalized_ids


def _drop_excluded_ids(ids, exclude_ids):
    exclude_set = set(exclude_ids)
    return [bearing_id for bearing_id in ids if bearing_id not in exclude_set]


def _make_equal_band_ranges(sampling_rate, n_bands, low_freq_start):
    nyquist = sampling_rate / 2.0
    edges = np.linspace(0.0, nyquist, n_bands + 1)
    bands = []
    for band_idx in range(n_bands):
        low = edges[band_idx]
        high = edges[band_idx + 1]
        if band_idx == 0:
            low = max(float(low_freq_start), np.finfo(float).eps)
        bands.append((float(low), float(high)))
    return bands


def _design_bandpass_sos(low, high, sampling_rate, filter_order):
    nyquist = sampling_rate / 2.0
    safe_low = max(float(low), np.finfo(float).eps)
    safe_high = min(float(high), nyquist * (1.0 - 1e-6))
    if not 0 < safe_low < safe_high < nyquist:
        raise ValueError(f"Invalid bandpass range: low={low}, high={high}, sampling_rate={sampling_rate}")
    return butter(filter_order, [safe_low, safe_high], btype='bandpass', fs=sampling_rate, output='sos')


def _load_vibration_1(mat_path):
    mat = scipy.io.loadmat(mat_path)
    key = os.path.splitext(os.path.basename(mat_path))[0]
    if key not in mat:
        raise KeyError(f"{mat_path}에서 MATLAB key '{key}'를 찾을 수 없습니다.")

    for item in mat[key][0][0]:
        if 'Name' in str(item.dtype) and 'vibration_1' in item[0]['Name']:
            idx = np.argwhere(item[0]['Name'] == 'vibration_1')
            signal = item[0]['Data'][idx][0][0][0]
            return np.asarray(signal, dtype=np.float64).reshape(-1)
    raise KeyError(f"{mat_path}에서 vibration_1 신호를 찾을 수 없습니다.")


def _filter_signal(signal, sos_bank):
    return np.stack([sosfiltfilt(sos, signal).astype(np.float32) for sos in sos_bank], axis=0)


def _fit_band_scaler(file_tuple_list, sos_bank, window_size, stride_size):
    n_bands = len(sos_bank)
    band_sum = np.zeros(n_bands, dtype=np.float64)
    band_sumsq = np.zeros(n_bands, dtype=np.float64)
    band_count = 0

    for folder_path, filename in file_tuple_list:
        signal = _load_vibration_1(os.path.join(folder_path, filename))
        band_signal = _filter_signal(signal, sos_bank)
        for start in range(0, len(signal) - window_size + 1, stride_size):
            window = band_signal[:, start:start + window_size].astype(np.float64)
            band_sum += window.sum(axis=1)
            band_sumsq += np.square(window).sum(axis=1)
            band_count += window_size

    if band_count == 0:
        raise ValueError("❌ train split에서 scaler를 fit할 수 있는 bandpass window가 없습니다.")

    band_mean = band_sum / band_count
    band_var = np.maximum((band_sumsq / band_count) - np.square(band_mean), 1e-12)
    band_std = np.sqrt(band_var)
    return band_mean.astype(np.float32), band_std.astype(np.float32)


def _extract_bandpass_windows(file_tuple_list, sos_bank, window_size, stride_size, band_mean, band_std, n_bands):
    extracted_windows = []
    extracted_ids = []

    for folder_path, filename in file_tuple_list:
        bearing_id = extract_bearing_id(filename)
        signal = _load_vibration_1(os.path.join(folder_path, filename))
        band_signal = _filter_signal(signal, sos_bank)
        normalized_signal = (band_signal - band_mean[:, np.newaxis]) / band_std[:, np.newaxis]

        for start in range(0, len(signal) - window_size + 1, stride_size):
            extracted_windows.append(normalized_signal[:, start:start + window_size].astype(np.float32))
            extracted_ids.append(bearing_id)

    if len(extracted_windows) == 0:
        return np.empty((0, n_bands, window_size), dtype=np.float32), np.array([], dtype=str)
    return np.stack(extracted_windows, axis=0), np.array(extracted_ids)


def _concat_or_empty(arrays, n_bands, window_size):
    non_empty = [array for array in arrays if len(array) > 0]
    if len(non_empty) == 0:
        return np.empty((0, n_bands, window_size), dtype=np.float32)
    return np.concatenate(non_empty, axis=0)


def _print_paderborn_config(loads, sampling_rate, window_size, stride_size, n_bands, bands,
                            n_sensor, train_count, val_count, test_count,
                            train_ids, val_ids, test_norm_ids, exclude_ids):
    band_ranges = ', '.join(f'[{low:.1f}, {high:.1f}] Hz' for low, high in bands)
    print('📈 [Paderborn OCC]')
    print(f'   - target settings: {loads}')
    print('   - input representation: bandpass_vib')
    print(f'   - sampling rate: {sampling_rate} Hz')
    print(f'   - raw window size: {window_size}')
    print(f'   - stride: {stride_size}')
    print(f'   - number of bands: {n_bands}')
    print(f'   - band frequency ranges: {band_ranges}')
    print(f'   - n_sensor: {n_sensor}')
    print(f'   - sequence length: {window_size}')
    print(f'   - train/val/test window counts: {train_count}/{val_count}/{test_count}')
    print(f'   - train IDs: {train_ids}')
    print(f'   - val IDs: {val_ids}')
    print(f'   - test normal IDs: {test_norm_ids}')
    print(f'   - exclude IDs: {exclude_ids}')


if __name__ == '__main__':
    train_l, val_l, test_l, ns = loader_Paderborn_OCC(
        root='/home/dayoon/DCP/Data/Paderborn',
        loads=['N15_M07_F10', 'N15_M01_F10'],
        window_size=2048,
        stride_size=1024,
        exclude_ids=['K006'],
    )
    batch_x, batch_y, batch_idx = next(iter(train_l))
    print(batch_x.shape, batch_y.shape, batch_idx.shape, ns)
