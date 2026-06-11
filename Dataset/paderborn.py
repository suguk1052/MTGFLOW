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

SENSOR_MODE_MAP = {
    "vib": ["vibration_1"],
    "mcs": ["phase_current_1", "phase_current_2"],
    "vib_mcs": ["vibration_1", "phase_current_1", "phase_current_2"],
}


def get_sensor_names(sensor_mode):
    if sensor_mode not in SENSOR_MODE_MAP:
        raise ValueError(
            f"Unsupported sensor_mode: {sensor_mode}. "
            f"Choose one of {list(SENSOR_MODE_MAP.keys())}."
        )
    return SENSOR_MODE_MAP[sensor_mode]


def _to_sensor_name(name_value):
    name_array = np.asarray(name_value).squeeze()
    if isinstance(name_array, np.ndarray):
        if name_array.size == 0:
            return ""
        if name_array.size == 1:
            name_array = name_array.item()
        else:
            name_array = "".join(str(item) for item in name_array.flatten())
    if isinstance(name_array, bytes):
        return name_array.decode()
    return str(name_array)


def extract_signal_from_mat(mat, key, sensor_name):
    y_array = mat[key][0, 0]["Y"]

    for i in range(y_array.shape[1]):
        sensor_room = y_array[0, i]
        name = _to_sensor_name(sensor_room["Name"][0])

        if name == sensor_name:
            return sensor_room["Data"].flatten()

    raise ValueError(f"Sensor {sensor_name} not found in {key}")


def load_multichannel_signal(mat_path, sensor_names):
    mat = scipy.io.loadmat(mat_path)
    key = os.path.basename(mat_path).replace(".mat", "")

    signals = [extract_signal_from_mat(mat, key, name) for name in sensor_names]

    lengths = [len(sig) for sig in signals]
    if len(set(lengths)) != 1:
        min_len = min(lengths)
        signals = [sig[:min_len] for sig in signals]

    return np.stack(signals, axis=1)

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
        # [Batch, Length, 1] -> MTGFlow 규격 맞춤 [1, Length, 1]
        window_data = self.windows[index].reshape([self.window_size, -1, 1])
        return torch.FloatTensor(window_data).transpose(0, 1), self.label[index], index


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
                         sensor_mode="vib"):
    """
    여러 세팅 폴더를 동시에 읽어와 통합 학습/추론이 가능한 OCC 데이터 로더
    예시: loads=['N15_M07_F10', 'N15_M01_F10'] 세팅 0과 세팅 2 동시 타겟팅
    """
    sensor_names = get_sensor_names(sensor_mode)
    n_sensor = len(sensor_names)
    
    def normalize_id_list(ids):
        normalized = []
        for item in ids or []:
            normalized.extend(part.strip() for part in str(item).split(',') if part.strip())
        return normalized

    train_ids = set(normalize_id_list(train_ids))
    val_ids = set(normalize_id_list(val_ids))
    test_norm_ids = set(normalize_id_list(test_norm_ids))
    exclude_ids = set(normalize_id_list(exclude_ids))

    def extract_bearing_id(filename):
        match = re.search(r'(K[A-Z]?\d{2,3})(?:_|\.)', filename)
        return match.group(1) if match else filename.replace('.mat', '')

    # 여러 도메인의 파일들을 추적하기 위해 (폴더절대경로, 파일명) 튜플 형태로 수집 리스트 선언
    train_file_tuples = []
    val_file_tuples = []
    test_normal_file_tuples = []
    test_fault_file_tuples = []

    # 🚀 Step 1: 지정된 모든 하중(운전 조건) 폴더를 순회하며 파일 분류 수집
    for load_setting in loads:
        setting_path = os.path.join(root, load_setting)
        if not os.path.exists(setting_path):
            print(f"⚠️ 경고: 해당 운전 조건 폴더가 경로에 없어 건너뜁니다: {setting_path}")
            continue
            
        filenames = [f for f in os.listdir(setting_path) if f.endswith('.mat')]
        
        for f in filenames:
            bearing_id = extract_bearing_id(f)
            if bearing_id in exclude_ids:
                continue

            # 튜플 구조로 (실제폴더경로, 파일명) 저장하여 물리적 위치 분리 보존
            file_info = (setting_path, f)
            
            if bearing_id in train_ids:
                train_file_tuples.append(file_info)
            elif bearing_id in val_ids:
                val_file_tuples.append(file_info)
            elif bearing_id in test_norm_ids:
                test_normal_file_tuples.append(file_info)
            else:
                # 지정된 정상 계열 외의 모든 실제 결함 파일들 수집
                test_fault_file_tuples.append(file_info)

    if len(train_file_tuples) == 0:
        raise ValueError("❌ 지정된 조건에 맞는 학습 데이터 파일(.mat)을 찾을 수 없습니다.")

    # 🚀 Step 2: 다중 도메인 통합 StandardScaler 피팅 (Data Leakage 절대 방지)
    # 여러 하중 환경의 정상 신호를 모두 모아서 단 하나의 글로벌 스케일러를 학습시킵니다.
    raw_train_signals = [
        load_multichannel_signal(os.path.join(folder_path, f), sensor_names)
        for folder_path, f in train_file_tuples
    ]
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate(raw_train_signals, axis=0))

    # 🚀 Step 3: 파일 경계면 브레이크 없이 윈도우를 추출하는 내부 헬퍼 함수
    def extract_scaled_windows(file_tuple_list):
        extracted_windows = []
        extracted_ids = []
        for folder_path, f in file_tuple_list:
            bearing_id = extract_bearing_id(f)
            sig = load_multichannel_signal(os.path.join(folder_path, f), sensor_names)

            # 통합 스케일러로 채널별 정규화 매핑
            scaled_sig = scaler.transform(sig)

            # 파일 단위 경계면을 침범하지 않는 독립 슬라이딩
            start = 0
            while start + window_size <= len(scaled_sig):
                extracted_windows.append(scaled_sig[start:start + window_size, :])
                extracted_ids.append(bearing_id)
                start += stride_size
                
        if not extracted_windows:
            return np.empty((0, window_size, n_sensor)), np.array([], dtype=str)
        return np.array(extracted_windows), np.array(extracted_ids)

    # 🚀 Step 4: 멀티 도메인 데이터셋 윈도우 가공 및 빌딩
    train_x, train_ids_per_window = extract_scaled_windows(train_file_tuples)
    val_x, val_ids_per_window = extract_scaled_windows(val_file_tuples)
    test_norm_x, test_norm_ids_per_window = extract_scaled_windows(test_normal_file_tuples)
    test_fault_x, test_fault_ids_per_window = extract_scaled_windows(test_fault_file_tuples)

    test_x = np.concatenate([test_norm_x, test_fault_x], axis=0)
    test_ids_per_window = np.concatenate([test_norm_ids_per_window, test_fault_ids_per_window], axis=0)
    
    # 이진 라벨 정의 (정상 0, 이상 1)
    train_y = np.zeros(len(train_x))
    val_y = np.zeros(len(val_x))
    test_y = np.array([0] * len(test_norm_x) + [1] * len(test_fault_x))

    print(f'📈 [Multi-Domain OCC] Target Settings: {loads}')
    print(f'   - Sensor Mode: {sensor_mode}')
    print(f'   - Sensor Names: {sensor_names}')
    print(f'   - n_sensor: {n_sensor}')
    if exclude_ids:
        print(f'   - Excluded Bearing IDs: {sorted(exclude_ids)}')
    print(f'   - Total Train Windows: {len(train_x)} | Val Windows: {len(val_x)} | Test Windows: {len(test_x)}')

    # 파이토치 데이터로더 패킹 및 반환
    train_loader = DataLoader(Paderborn_dataset(train_x, train_y, window_size, train_ids_per_window), batch_size=batch_size, shuffle=not label)
    val_loader = DataLoader(Paderborn_dataset(val_x, val_y, window_size, val_ids_per_window), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Paderborn_dataset(test_x, test_y, window_size, test_ids_per_window), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Paderborn loader smoke test')
    parser.add_argument('--root', default='/home/dayoon/DCP/Data/Paderborn')
    parser.add_argument('--load_setting', nargs='+', default=['N15_M07_F10'])
    parser.add_argument('--sensor_mode', choices=list(SENSOR_MODE_MAP.keys()), default='mcs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--window_size', type=int, default=2048)
    parser.add_argument('--stride_size', type=int, default=1024)
    smoke_args = parser.parse_args()

    train_l, _, _, ns = loader_Paderborn_OCC(
        root=smoke_args.root,
        loads=smoke_args.load_setting,
        batch_size=smoke_args.batch_size,
        window_size=smoke_args.window_size,
        stride_size=smoke_args.stride_size,
        sensor_mode=smoke_args.sensor_mode,
    )
    batch_x, _, _ = next(iter(train_l))
    print(f"sensor_mode: {smoke_args.sensor_mode}")
    print(f"sensor_names: {get_sensor_names(smoke_args.sensor_mode)}")
    print(f"n_sensor: {ns}")
    print(f"batch x shape: {batch_x.shape}")
