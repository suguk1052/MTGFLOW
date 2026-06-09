# ==============================================================================
# Modified by Dayoon (2026) for Multi-Load / Multi-Domain TSAD Experiments
# Description: Upgraded Paderborn Loader to support FFT vibration-only inputs.
# ==============================================================================

import os
import numpy as np
import scipy.io
import re
import torch
from torch.utils.data import Dataset, DataLoader


PADERBORN_INPUT_MODE = "FFT vibration"
PADERBORN_SIGNAL_NAME = "vibration_1"
PADERBORN_N_FFT_BANDS = 8


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
            # FFT input is already [entity/channel, sequence_length].
            window_data = window[:, :, np.newaxis]
        else:
            # Backward-compatible raw waveform path: [length] -> [1, length, 1].
            window_data = window.reshape([self.window_size, -1, 1]).transpose(1, 0, 2)
        return torch.FloatTensor(window_data), self.label[index], index


def _extract_bearing_id(filename):
    match = re.search(r'(K[A-Z]?\d{2,3})(?:_|\.)', filename)
    return match.group(1) if match else filename.replace('.mat', '')


def _read_vibration_1_signal(mat_path):
    mat = scipy.io.loadmat(mat_path)
    key = os.path.basename(mat_path).replace('.mat', '')
    for channel_group in mat[key][0][0]:
        if 'Name' in str(channel_group.dtype) and PADERBORN_SIGNAL_NAME in channel_group[0]['Name']:
            idx = np.argwhere(channel_group[0]['Name'] == PADERBORN_SIGNAL_NAME)
            return channel_group[0]['Data'][idx][0][0][0].reshape(-1)
    return None


def _raw_window_to_fft_bands(raw_window, n_fft_bands=PADERBORN_N_FFT_BANDS):
    magnitude = np.abs(np.fft.rfft(raw_window))[1:]
    magnitude = np.log1p(magnitude)

    if len(magnitude) % n_fft_bands != 0:
        raise ValueError(
            f"FFT bin count after DC removal ({len(magnitude)}) must be divisible by "
            f"n_fft_bands ({n_fft_bands})."
        )

    band_length = len(magnitude) // n_fft_bands
    return magnitude.reshape(n_fft_bands, band_length)


def _extract_fft_windows(file_tuple_list, raw_window_size, stride_size, n_fft_bands):
    extracted_windows = []
    extracted_ids = []
    for folder_path, filename in file_tuple_list:
        bearing_id = _extract_bearing_id(filename)
        sig = _read_vibration_1_signal(os.path.join(folder_path, filename))
        if sig is None:
            print(f"⚠️ 경고: {filename}에서 {PADERBORN_SIGNAL_NAME} 채널을 찾지 못해 건너뜁니다.")
            continue

        start = 0
        while start + raw_window_size <= len(sig):
            raw_window = sig[start:start + raw_window_size]
            extracted_windows.append(_raw_window_to_fft_bands(raw_window, n_fft_bands))
            extracted_ids.append(bearing_id)
            start += stride_size

    if len(extracted_windows) == 0:
        return np.empty((0, n_fft_bands, raw_window_size // (2 * n_fft_bands))), np.array(extracted_ids)
    return np.asarray(extracted_windows, dtype=np.float32), np.asarray(extracted_ids)


def _fit_channelwise_scaler(train_fft_windows):
    if len(train_fft_windows) == 0:
        raise ValueError("❌ train split의 FFT feature가 비어 있어 scaler를 계산할 수 없습니다.")
    mean = train_fft_windows.mean(axis=(0, 2), keepdims=True)
    std = train_fft_windows.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_channelwise_scaler(windows, mean, std):
    if len(windows) == 0:
        return windows.astype(np.float32)
    return ((windows - mean) / std).astype(np.float32)


def _concat_or_empty(arrays, shape_tail):
    non_empty = [arr for arr in arrays if len(arr) > 0]
    if non_empty:
        return np.concatenate(non_empty, axis=0)
    return np.empty((0, *shape_tail), dtype=np.float32)


def loader_Paderborn_OCC(root="/home/dayoon/DCP/Data/Paderborn",
                         loads=["N15_M07_F10"],
                         batch_size=64,
                         window_size=128,
                         stride_size=1024,
                         label=False,
                         train_ids=['K001', 'K002', 'K003'],
                         val_ids=['K004'],
                         test_norm_ids=['K005', 'K006'],
                         raw_window_size=2048,
                         n_fft_bands=PADERBORN_N_FFT_BANDS):
    """
    여러 세팅 폴더를 동시에 읽어와 통합 학습/추론이 가능한 OCC 데이터 로더.

    Paderborn은 vibration_1 raw waveform을 파일 단위 sliding window로 자른 뒤,
    window마다 rFFT magnitude -> DC 제거 -> log1p -> 균등 frequency band 분할을 적용한다.
    최종 입력 shape은 [n_fft_bands, sequence_length, 1]이다.
    """
    fft_bin_count = raw_window_size // 2
    if raw_window_size % 2 != 0:
        raise ValueError("raw_window_size는 np.fft.rfft 후 DC 제거 bin 수 계산을 위해 짝수여야 합니다.")
    if fft_bin_count % n_fft_bands != 0:
        raise ValueError(
            f"raw_window_size={raw_window_size}에서 DC 제거 후 bin 수 {fft_bin_count}가 "
            f"n_fft_bands={n_fft_bands}로 나누어떨어지지 않습니다."
        )

    sequence_length = fft_bin_count // n_fft_bands
    if window_size != sequence_length:
        print(
            f"⚠️ Paderborn FFT 입력에서는 model window_size를 {sequence_length}로 사용합니다 "
            f"(전달값: {window_size})."
        )
        window_size = sequence_length

    train_file_tuples = []
    val_file_tuples = []
    test_normal_file_tuples = []
    test_fault_file_tuples = []

    for load_setting in loads:
        setting_path = os.path.join(root, load_setting)
        if not os.path.exists(setting_path):
            print(f"⚠️ 경고: 해당 운전 조건 폴더가 경로에 없어 건너뜁니다: {setting_path}")
            continue

        filenames = [f for f in os.listdir(setting_path) if f.endswith('.mat')]

        for filename in filenames:
            file_info = (setting_path, filename)

            if any(bid in filename for bid in train_ids):
                train_file_tuples.append(file_info)
            elif any(bid in filename for bid in val_ids):
                val_file_tuples.append(file_info)
            elif any(bid in filename for bid in test_norm_ids):
                test_normal_file_tuples.append(file_info)
            else:
                test_fault_file_tuples.append(file_info)

    if len(train_file_tuples) == 0:
        raise ValueError("❌ 지정된 조건에 맞는 학습 데이터 파일(.mat)을 찾을 수 없습니다.")

    train_fft, train_ids_per_window = _extract_fft_windows(
        train_file_tuples, raw_window_size, stride_size, n_fft_bands
    )
    val_fft, val_ids_per_window = _extract_fft_windows(
        val_file_tuples, raw_window_size, stride_size, n_fft_bands
    )
    test_norm_fft, test_norm_ids_per_window = _extract_fft_windows(
        test_normal_file_tuples, raw_window_size, stride_size, n_fft_bands
    )
    test_fault_fft, test_fault_ids_per_window = _extract_fft_windows(
        test_fault_file_tuples, raw_window_size, stride_size, n_fft_bands
    )

    scaler_mean, scaler_std = _fit_channelwise_scaler(train_fft)
    train_x = _apply_channelwise_scaler(train_fft, scaler_mean, scaler_std)
    val_x = _apply_channelwise_scaler(val_fft, scaler_mean, scaler_std)
    test_norm_x = _apply_channelwise_scaler(test_norm_fft, scaler_mean, scaler_std)
    test_fault_x = _apply_channelwise_scaler(test_fault_fft, scaler_mean, scaler_std)

    shape_tail = (n_fft_bands, sequence_length)
    test_x = _concat_or_empty([test_norm_x, test_fault_x], shape_tail)
    test_ids_per_window = np.concatenate([test_norm_ids_per_window, test_fault_ids_per_window], axis=0)

    train_y = np.zeros(len(train_x))
    val_y = np.zeros(len(val_x))
    test_y = np.array([0] * len(test_norm_x) + [1] * len(test_fault_x))

    n_sensor = n_fft_bands

    print("📈 [Paderborn OCC Loader]")
    print(f"   - Load settings: {loads}")
    print(f"   - Input mode: {PADERBORN_INPUT_MODE} ({PADERBORN_SIGNAL_NAME} only)")
    print(f"   - Raw window size: {raw_window_size}")
    print(f"   - Stride: {stride_size}")
    print(f"   - FFT bands: {n_fft_bands}")
    print(f"   - Final n_sensor: {n_sensor}")
    print(f"   - Final sequence length: {sequence_length}")
    print(f"   - Train/Val/Test windows: {len(train_x)} / {len(val_x)} / {len(test_x)}")
    print(f"   - Train normal bearing IDs: {train_ids}")
    print(f"   - Val normal bearing IDs: {val_ids}")
    print(f"   - Test normal bearing IDs: {test_norm_ids}")

    train_loader = DataLoader(Paderborn_dataset(train_x, train_y, window_size, train_ids_per_window), batch_size=batch_size, shuffle=not label)
    val_loader = DataLoader(Paderborn_dataset(val_x, val_y, window_size, val_ids_per_window), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Paderborn_dataset(test_x, test_y, window_size, test_ids_per_window), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor


if __name__ == '__main__':
    train_l, val_l, test_l, ns = loader_Paderborn_OCC(
        root='/home/dayoon/DCP/Data/Paderborn',
        loads=['N15_M07_F10', 'N15_M01_F10'],
        raw_window_size=2048,
        stride_size=1024,
        window_size=128,
    )
    xb, yb, ib = next(iter(train_l))
    print(f"batch shape: {xb.shape}, n_sensor: {ns}")
