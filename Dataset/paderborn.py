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
                         test_norm_ids=['K005', 'K006']):
    """
    여러 세팅 폴더를 동시에 읽어와 통합 학습/추론이 가능한 OCC 데이터 로더
    예시: loads=['N15_M07_F10', 'N15_M01_F10'] 세팅 0과 세팅 2 동시 타겟팅
    """
    
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
            # 튜플 구조로 (실제폴더경로, 파일명) 저장하여 물리적 위치 분리 보존
            file_info = (setting_path, f)
            
            if any(bid in f for bid in train_ids):
                train_file_tuples.append(file_info)
            elif any(bid in f for bid in val_ids):
                val_file_tuples.append(file_info)
            elif any(bid in f for bid in test_norm_ids):
                test_normal_file_tuples.append(file_info)
            else:
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
            if 'Name' in str(j.dtype) and 'vibration_1' in j[0]['Name']:
                idx = np.argwhere(j[0]['Name'] == 'vibration_1')
                raw_train_signals.append(j[0]['Data'][idx][0][0][0])
                break
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate(raw_train_signals).reshape(-1, 1))

    def extract_bearing_id(filename):
        match = re.search(r'(K[A-Z]?\d{2,3})(?:_|\.)', filename)
        return match.group(1) if match else filename.replace('.mat', '')

    # 🚀 Step 3: 파일 경계면 브레이크 없이 윈도우를 추출하는 내부 헬퍼 함수
    def extract_scaled_windows(file_tuple_list):
        extracted_windows = []
        extracted_ids = []
        for folder_path, f in file_tuple_list:
            bearing_id = extract_bearing_id(f)
            mat = scipy.io.loadmat(os.path.join(folder_path, f))
            key = f.replace('.mat', '')
            sig = None
            for j in mat[key][0][0]:
                if 'Name' in str(j.dtype) and 'vibration_1' in j[0]['Name']:
                    idx = np.argwhere(j[0]['Name'] == 'vibration_1')
                    sig = j[0]['Data'][idx][0][0][0]
                    break
            if sig is None:
                continue

            # 통합 스케일러로 정규화 매핑 후 1차원 플래튼
            scaled_sig = scaler.transform(sig.reshape(-1, 1)).flatten()

            # 파일 단위 경계면을 침범하지 않는 독립 슬라이딩
            start = 0
            while start + window_size <= len(scaled_sig):
                extracted_windows.append(scaled_sig[start:start + window_size])
                extracted_ids.append(bearing_id)
                start += stride_size
                
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

    n_sensor = 1 

    print(f'📈 [Multi-Domain OCC] Target Settings: {loads}')
    print(f'   - Total Train Windows: {len(train_x)} | Val Windows: {len(val_x)} | Test Windows: {len(test_x)}')

    # 파이토치 데이터로더 패킹 및 반환
    train_loader = DataLoader(Paderborn_dataset(train_x, train_y, window_size, train_ids_per_window), batch_size=batch_size, shuffle=not label)
    val_loader = DataLoader(Paderborn_dataset(val_x, val_y, window_size, val_ids_per_window), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Paderborn_dataset(test_x, test_y, window_size, test_ids_per_window), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor


if __name__ == '__main__':
    # 로컬 가동 및 차원 디버깅 테스트용 플래그
    # 세팅 0(N15_M07_F10)과 세팅 2(N15_M01_F10) 데이터를 동시에 묶어서 로드하는 예시
    train_l, val_l, test_l, ns = loader_Paderborn_OCC(
        file_path='/home/dayoon/DCP/Data/Paderborn',
        loads=['N15_M07_F10', 'N15_M01_F10'], # 🚀 두 개 폴더 동시 지정 테스트
        window_size=2048,
        stride_size=1024
    )