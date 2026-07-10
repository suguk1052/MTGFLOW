# ==============================================================================
# TODO.md §1 진단 스크립트 A
# measured 운행값 조건의 (1) train 기준 |z-score|/coverage, (2) train_meta_std
# 팽창(가설 A — 통계 오염 버전) 여부를 4 LOSO x 6 LONO = 24개 조합에서 계산한다.
# 새 학습 없음, GPU 불필요. Dataset/paderborn.py의 프로덕션 로더 함수를 그대로
# 재사용하며(재구현 금지), z-score 값은 loader_Paderborn_OCC()가 실제 학습/평가에서
# 쓰는 정규화 결과를 그대로 읽는다.
# ==============================================================================

import contextlib
import io
import json
import os
import re
import sys

import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Dataset.paderborn import (
    extract_signal_from_mat,
    load_measured_operational_signals,
    loader_Paderborn_OCC,
    summarize_measured_meta,
)

ROOT = "/home/dayoon/DCP/Data/Paderborn"
WINDOW_SIZE = 2048
STRIDE_SIZE = 1024
SENSOR_MODE = "vibration_1"
MEASURED_META_STATS = "meanstd"
META_DIMS = ["speed_mean", "speed_std", "torque_mean", "torque_std", "force_mean", "force_std"]
Z_THRESHOLD = 2.0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "Paderborn", "diag_measured_gate")

LOSO_SPLITS = {
    "123to0": dict(train_loads=["N09_M07_F10", "N15_M01_F10", "N15_M07_F04"], test_loads=["N15_M07_F10"], held_out="N15_M07_F10"),
    "023to1": dict(train_loads=["N15_M07_F10", "N15_M01_F10", "N15_M07_F04"], test_loads=["N09_M07_F10"], held_out="N09_M07_F10"),
    "013to2": dict(train_loads=["N15_M07_F10", "N09_M07_F10", "N15_M07_F04"], test_loads=["N15_M01_F10"], held_out="N15_M01_F10"),
    "012to3": dict(train_loads=["N15_M07_F10", "N09_M07_F10", "N15_M01_F10"], test_loads=["N15_M07_F04"], held_out="N15_M07_F04"),
}

# MTGFLOW/CLAUDE.md §4 LONO 표와 동일
LONO_SPLITS = {
    1: dict(train_ids=["K003", "K004", "K005", "K006"], val_ids=["K002"], test_norm_ids=["K001"]),
    2: dict(train_ids=["K001", "K004", "K005", "K006"], val_ids=["K003"], test_norm_ids=["K002"]),
    3: dict(train_ids=["K001", "K002", "K005", "K006"], val_ids=["K004"], test_norm_ids=["K003"]),
    4: dict(train_ids=["K001", "K002", "K003", "K006"], val_ids=["K005"], test_norm_ids=["K004"]),
    5: dict(train_ids=["K001", "K002", "K003", "K004"], val_ids=["K006"], test_norm_ids=["K005"]),
    6: dict(train_ids=["K002", "K003", "K004", "K005"], val_ids=["K001"], test_norm_ids=["K006"]),
}

# Dataset/paderborn.py의 extract_bearing_id()와 동일한 규칙(재사용, 로더 내부라 직접 import 불가)
BEARING_ID_RE = re.compile(r"(K[A-Z]?\d{2,3})(?:_|\.)")


def extract_bearing_id(filename):
    match = BEARING_ID_RE.search(filename)
    return match.group(1) if match else filename.replace(".mat", "")


def raw_train_meta(train_loads, train_ids):
    """train 파일들의 window-level measured meta를 정규화 전 상태로 직접 계산한다.
    (가설 A 통계-오염 체크: train_meta_std가 세팅1 포함 여부에 따라 팽창하는지 보기 위함.)"""
    metas = []
    for load_setting in train_loads:
        setting_path = os.path.join(ROOT, load_setting)
        for fname in sorted(os.listdir(setting_path)):
            if not fname.endswith(".mat"):
                continue
            if extract_bearing_id(fname) not in train_ids:
                continue
            mat_path = os.path.join(setting_path, fname)
            measured_signals = load_measured_operational_signals(mat_path)
            mat = scipy.io.loadmat(mat_path)
            key = fname.replace(".mat", "")
            sig = extract_signal_from_mat(mat, key, SENSOR_MODE)
            start = 0
            while start + WINDOW_SIZE <= len(sig):
                metas.append(summarize_measured_meta(
                    measured_signals, start, start + WINDOW_SIZE, MEASURED_META_STATS,
                ))
                start += STRIDE_SIZE
    return np.array(metas, dtype=np.float32)


def run_one(loso_name, loso, lono_idx, lono):
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, test_loader, _ = loader_Paderborn_OCC(
            root=ROOT,
            train_loads=loso["train_loads"],
            test_loads=loso["test_loads"],
            train_ids=lono["train_ids"],
            val_ids=lono["val_ids"],
            test_norm_ids=lono["test_norm_ids"],
            window_size=WINDOW_SIZE,
            stride_size=STRIDE_SIZE,
            sensor_mode=SENSOR_MODE,
            meta_source="measured",
            measured_meta_stats=MEASURED_META_STATS,
        )

    dataset = test_loader.dataset
    normal_mask = dataset.label == 0
    z = dataset.metas[normal_mask]  # loader가 이미 train 기준으로 z-score한 값

    train_meta = raw_train_meta(loso["train_loads"], set(lono["train_ids"]))

    abs_z = np.abs(z)
    row = {
        "loso_split": loso_name,
        "held_out_setting": loso["held_out"],
        "lono_idx": lono_idx,
        "n_test_normal_windows": int(normal_mask.sum()),
        "n_train_windows": int(len(train_meta)),
        "z_abs_mean": {dim: float(abs_z[:, i].mean()) for i, dim in enumerate(META_DIMS)},
        "z_abs_median": {dim: float(np.median(abs_z[:, i])) for i, dim in enumerate(META_DIMS)},
        f"coverage_pct_z_gt_{Z_THRESHOLD}": {
            dim: float((abs_z[:, i] > Z_THRESHOLD).mean() * 100) for i, dim in enumerate(META_DIMS)
        },
        "train_meta_std": {dim: float(train_meta[:, i].std()) for i, dim in enumerate(META_DIMS)},
    }
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for loso_name, loso in LOSO_SPLITS.items():
        for lono_idx, lono in LONO_SPLITS.items():
            print(f"[{loso_name} / LONO-{lono_idx}] 계산 중...", file=sys.stderr)
            rows.append(run_one(loso_name, loso, lono_idx, lono))

    out_path = os.path.join(OUT_DIR, "zscore_stats.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {out_path}")

    print("\n=== LOSO split별 요약 (6 LONO 평균) ===")
    for loso_name in LOSO_SPLITS:
        split_rows = [r for r in rows if r["loso_split"] == loso_name]
        speed_mean_z = np.mean([r["z_abs_mean"]["speed_mean"] for r in split_rows])
        speed_std_z = np.mean([r["z_abs_mean"]["speed_std"] for r in split_rows])
        cov_key = f"coverage_pct_z_gt_{Z_THRESHOLD}"
        speed_cov = np.mean([r[cov_key]["speed_mean"] for r in split_rows])
        train_speed_std = np.mean([r["train_meta_std"]["speed_mean"] for r in split_rows])
        setting1_in_train = "N09_M07_F10" in LOSO_SPLITS[loso_name]["train_loads"]
        print(
            f"{loso_name} (held_out={LOSO_SPLITS[loso_name]['held_out']}, "
            f"세팅1 in train={setting1_in_train}): "
            f"|z|_speed_mean={speed_mean_z:.3f}, |z|_speed_std={speed_std_z:.3f}, "
            f"coverage(|z|>{Z_THRESHOLD})={speed_cov:.1f}%, train_speed_mean_std={train_speed_std:.3f}"
        )


if __name__ == "__main__":
    main()
