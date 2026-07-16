# ==============================================================================
# TODO.md §1 진단 스크립트 B
# 4개 세팅 각각의 정상(K001~K006) vibration_1이 "나머지 세팅 대비 정상 진동 분포를
# 실제로 얼마나 지배하는가"를 정량화한다 (가설 B 재료). 새 학습 없음, GPU 불필요.
#
# 계산 기준은 스크립트 A / 프로덕션 파이프라인과 통일: window_size=2048, stride=1024,
# StandardScaler(전체 정상 K001~K006 pooled fit) 적용 후 window별 feature 계산.
#
# feature 6종:
#   - rms/std/absmax/kurtosis: Dataset/K006_EDA.ipynb의 safe_stats() 공식 재사용
#   - wpd_node0_energy/clearance_factor: Data/paderborn_eda.ipynb의
#     get_wpd_energies()/calculate_clearance_factor() 공식만 재사용
#     (입력은 원본의 200ms raw 세그먼트가 아니라 위 2048-window/정규화 신호로 재구현 — 기준 통일)
#
# 주 지표(연구노트 0703 방식의 정량화, 우선): boxplot의 median/IQR을 그대로 정량화
#   - median_shift_norm = (median_i - median_rest) / IQR_rest
#   - box_nonoverlap_pct = setting_i의 IQR 박스가 rest의 IQR 박스와 겹치지 않는 비율
# 보조 지표(추가 제안, 대체 아님): Cohen's d, Wasserstein distance
# ==============================================================================

import json
import os
import sys

import numpy as np
import pywt
import scipy.io
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

# analysis/ 아래로 옮겨졌으므로 MTGFLOW 루트(analysis/의 부모)를 sys.path에 넣어야 analysis._common을 찾는다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from analysis._common import extract_bearing_id

ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, '..', 'Data', 'Paderborn'))
WINDOW_SIZE = 2048
STRIDE_SIZE = 1024
SENSOR_MODE = "vibration_1"
NORMAL_IDS = ["K001", "K002", "K003", "K004", "K005", "K006"]
SETTINGS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
FEATURES = ["rms", "std", "absmax", "kurtosis", "wpd_node0_energy", "clearance_factor"]

# 스크립트 위치(analysis/)와 무관하게 항상 MTGFLOW/results/... 를 가리키게 PROJECT_ROOT 기준으로 구성
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "Paderborn", "diag_measured_gate")


def load_vibration_1(mat_path):
    mat = scipy.io.loadmat(mat_path)
    key = os.path.basename(mat_path).replace(".mat", "")
    y_array = mat[key][0, 0]["Y"]
    for i in range(y_array.shape[1]):
        sensor_room = y_array[0, i]
        name = np.asarray(sensor_room["Name"][0]).squeeze()
        name_str = str(name.item()) if name.shape == () else "".join(str(c) for c in name.tolist())
        if name_str == SENSOR_MODE:
            return sensor_room["Data"].flatten().astype(np.float64)
    raise ValueError(f"{SENSOR_MODE} not found in {mat_path}")


# Data/paderborn_eda.ipynb의 get_wpd_energies() 공식 재사용(입력만 2048-window로 교체)
def wpd_node0_energy(window, wavelet="db4", level=3):
    wp = pywt.WaveletPacket(data=window, wavelet=wavelet, mode="symmetric", maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, order="freq")]
    energies = np.array([np.sum(wp[node].data ** 2) for node in nodes])
    return energies[0] / np.sum(energies)


# Data/paderborn_eda.ipynb의 calculate_clearance_factor() 공식 그대로 재사용
def clearance_factor(window):
    max_val = np.max(np.abs(window))
    mean_sqrt = np.mean(np.sqrt(np.abs(window))) ** 2
    return max_val / mean_sqrt


def window_features(window):
    return {
        "rms": float(np.sqrt(np.mean(window ** 2))),
        "std": float(np.std(window)),
        "absmax": float(np.max(np.abs(window))),
        "kurtosis": float(scipy_kurtosis(window)),
        "wpd_node0_energy": float(wpd_node0_energy(window)),
        "clearance_factor": float(clearance_factor(window)),
    }


def collect_setting_files(setting):
    setting_path = os.path.join(ROOT, setting)
    files = []
    for fname in sorted(os.listdir(setting_path)):
        if not fname.endswith(".mat"):
            continue
        if extract_bearing_id(fname) in NORMAL_IDS:
            files.append(os.path.join(setting_path, fname))
    return files


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("1) 전체 정상(K001~K006, 4개 세팅 pooled) vibration_1 로딩 + StandardScaler fit", file=sys.stderr)
    setting_files = {s: collect_setting_files(s) for s in SETTINGS}
    all_signals = {}
    pooled_for_scaler = []
    for setting, files in setting_files.items():
        for f in files:
            sig = load_vibration_1(f)
            all_signals[f] = sig
            pooled_for_scaler.append(sig.reshape(-1, 1))

    scaler = StandardScaler()
    scaler.fit(np.concatenate(pooled_for_scaler, axis=0))
    print(f"scaler.mean_={scaler.mean_[0]:.5f}, scaler.scale_={scaler.scale_[0]:.5f}", file=sys.stderr)

    print("2) 세팅별 window-level feature 계산", file=sys.stderr)
    setting_feature_values = {s: {feat: [] for feat in FEATURES} for s in SETTINGS}
    for setting, files in setting_files.items():
        for f in files:
            z = scaler.transform(all_signals[f].reshape(-1, 1)).flatten()
            start = 0
            while start + WINDOW_SIZE <= len(z):
                window = z[start:start + WINDOW_SIZE]
                feats = window_features(window)
                for feat in FEATURES:
                    setting_feature_values[setting][feat].append(feats[feat])
                start += STRIDE_SIZE
        n_windows = len(setting_feature_values[setting][FEATURES[0]])
        print(f"  {setting}: {len(files)} files, {n_windows} windows", file=sys.stderr)

    print("3) setting별 rest 대비 지배력 점수 계산 (주 지표: boxplot 정량화 / 보조 지표: Cohen's d, Wasserstein)", file=sys.stderr)
    rows = []
    for setting in SETTINGS:
        rest_settings = [s for s in SETTINGS if s != setting]
        row = {"setting": setting, "n_windows": len(setting_feature_values[setting][FEATURES[0]]), "features": {}}
        primary_scores = []
        for feat in FEATURES:
            vals_i = np.array(setting_feature_values[setting][feat])
            vals_rest = np.concatenate([np.array(setting_feature_values[s][feat]) for s in rest_settings])

            median_i, median_rest = np.median(vals_i), np.median(vals_rest)
            q1_i, q3_i = np.percentile(vals_i, [25, 75])
            q1_rest, q3_rest = np.percentile(vals_rest, [25, 75])
            iqr_rest = q3_rest - q1_rest
            iqr_i = q3_i - q1_i

            median_shift_norm = float((median_i - median_rest) / iqr_rest) if iqr_rest > 0 else float("nan")
            overlap = max(0.0, min(q3_i, q3_rest) - max(q1_i, q1_rest))
            box_nonoverlap_pct = float(1.0 - overlap / iqr_i) if iqr_i > 0 else float("nan")
            box_nonoverlap_pct = max(0.0, min(1.0, box_nonoverlap_pct)) if not np.isnan(box_nonoverlap_pct) else box_nonoverlap_pct

            pooled_std = np.sqrt((vals_i.std() ** 2 + vals_rest.std() ** 2) / 2)
            cohens_d = float((vals_i.mean() - vals_rest.mean()) / pooled_std) if pooled_std > 0 else float("nan")
            w_dist = float(wasserstein_distance(vals_i, vals_rest))

            row["features"][feat] = {
                "primary_median_shift_norm": median_shift_norm,
                "primary_box_nonoverlap_pct": box_nonoverlap_pct,
                "secondary_cohens_d": cohens_d,
                "secondary_wasserstein": w_dist,
            }
            if not np.isnan(median_shift_norm):
                primary_scores.append(abs(median_shift_norm))

        row["vibration_shift_score_primary"] = float(np.mean(primary_scores)) if primary_scores else float("nan")
        row["vibration_shift_score_secondary_cohens_d_mean_abs"] = float(
            np.mean([abs(row["features"][f]["secondary_cohens_d"]) for f in FEATURES])
        )
        rows.append(row)

    out_path = os.path.join(OUT_DIR, "vibration_shift.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {out_path}")

    print("\n=== 세팅별 지배력 점수 요약 (주 지표 = |median_shift_norm| 평균) ===")
    for row in sorted(rows, key=lambda r: -r["vibration_shift_score_primary"]):
        print(
            f"{row['setting']}: 주지표(median_shift_norm 평균 |.|)={row['vibration_shift_score_primary']:.3f}, "
            f"보조지표(Cohen's d 평균 |.|)={row['vibration_shift_score_secondary_cohens_d_mean_abs']:.3f}"
        )


if __name__ == "__main__":
    main()
