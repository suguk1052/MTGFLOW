"""B-1 (b) 6-log-band z-vector 이중 검증 (결정1-A). CPU, 학습 없음.

캐시 npz(p2_log_window_scores / UODS dump)에는 amp target 원값(rms_z)이 저장돼 있지 않고
S_amp 점수만 있다 → 원값 대조 불가. 대신 아래 두 방식으로 (b)가 제안 모델 amp target과
동일 z-vector임을 실증한다.

  검증1 (amp_normalize 불변성): 로더를 amp_normalize=False/True 두 번 호출해 .dataset.rms_z 의
    max|Δ| 를 측정(=0 기대). 학습 러너는 --amp_normalize(True)이고 B-1은 False지만, band target은
    paderborn.py:540-545 에서 amp_normalize 나눗셈 **이전**에 계산되므로 rms_z 는 불변.
  검증2 (독립 재현): amp_normalize=False 의 .dataset.windows(=정규화 전 window)에 paderborn 헬퍼
    compute_band_rms + log + train-normal z-score 를 **독립 적용**해 .dataset.rms_z 와 max|Δ| 비교(≈1e-6).

PU 몇 fold + UODS 몇 split 에 대해 표를 출력한다(리포트 붙여넣기용).
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from Dataset.paderborn import (  # noqa: E402
    loader_Paderborn_OCC, compute_band_boundaries, compute_band_rms,
)
from Dataset.uods import loader_UODS_OCC  # noqa: E402
from dump_B1_pu_window_scores import (  # noqa: E402
    DATA_ROOT, RESULTS_ROOT, load_fold_meta, B1_N_BANDS, B1_BAND_SCHEME,
)

UODS_DATA_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "Data", "UODS-VAFDC"))


def _loader_pu(meta, amp_normalize, bs=256):
    with contextlib.redirect_stdout(io.StringIO()):
        return loader_Paderborn_OCC(
            root=DATA_ROOT, train_loads=meta["train_load_setting"],
            test_loads=meta["test_load_setting"], train_ids=meta["train_ids"],
            val_ids=meta["val_ids"], test_norm_ids=meta["test_norm_ids"],
            exclude_ids=meta.get("exclude_ids", []),
            window_size=int(meta["window_size"]), stride_size=int(meta["stride_size"]),
            sensor_mode=meta["sensor_mode"], meta_source=meta.get("meta_source", "static"),
            measured_meta_stats=meta.get("measured_meta_stats", "meanstd"),
            amp_normalize=amp_normalize, amp_n_bands=B1_N_BANDS, amp_band_scheme=B1_BAND_SCHEME,
            amp_band_min_width=int(meta.get("amp_band_min_width", 4)),
            rms_eps=float(meta.get("rms_eps", 1e-8)), batch_size=bs)


def _loader_uods(manifest, amp_normalize, bs=256):
    if not os.path.isabs(manifest):
        manifest = os.path.join(PROJECT_ROOT, manifest)
    with contextlib.redirect_stdout(io.StringIO()):
        return loader_UODS_OCC(
            root=UODS_DATA_ROOT, manifest=manifest, batch_size=bs,
            window_size=2048, stride_size=1024, amp_normalize=amp_normalize,
            amp_n_bands=B1_N_BANDS, amp_band_scheme=B1_BAND_SCHEME, sampling_rate=42000)


def _independent_zvector(windows, tr_windows, window_size, rms_eps=1e-8):
    """정규화 전 window(1D ch0)에서 band z-vector 독립 재현 (paderborn.py:540-545,668-685 미러)."""
    F = window_size // 2 + 1
    boundaries = compute_band_boundaries(F, B1_N_BANDS, B1_BAND_SCHEME)
    def _band(ws):
        return np.stack([compute_band_rms(np.asarray(w, float), boundaries) for w in ws])  # (N,K)
    tr_band = _band(tr_windows)
    tr_logband = np.log(tr_band + rms_eps)
    band_mean = tr_logband.mean(axis=0)
    band_std = tr_logband.std(axis=0)
    x_logband = np.log(_band(windows) + rms_eps)
    return ((x_logband - band_mean) / (band_std + 1e-8)).astype(np.float32)


def verify_pu(split, lono):
    meta = load_fold_meta(split, lono)
    trF, vaF, teF, _ = _loader_pu(meta, amp_normalize=False)
    trT, vaT, teT, _ = _loader_pu(meta, amp_normalize=True)
    # 검증1: amp_normalize False vs True 의 rms_z
    d1 = max(float(np.abs(np.asarray(a.dataset.rms_z) - np.asarray(b.dataset.rms_z)).max())
             for a, b in ((trF, trT), (vaF, vaT), (teF, teT)))
    # 검증2: 독립 재현 vs 로더 rms_z (amp_normalize=False windows 사용)
    win = int(meta["window_size"])
    tr_w = np.asarray(trF.dataset.windows)
    z_te = _independent_zvector(np.asarray(teF.dataset.windows), tr_w, win)
    d2 = float(np.abs(z_te - np.asarray(teF.dataset.rms_z)).max())
    edges = getattr(teF.dataset, "amp_band_edges", None)
    return {"data": "PU", "fold": f"{split}_LONO{lono}", "edges": edges,
            "max|Δ|_ampnorm_invariance": d1, "max|Δ|_independent_rederive": d2,
            "n_te": int(len(teF.dataset.windows))}


def verify_uods(run_idx):
    manifest = os.path.join(RESULTS_ROOT.replace("Paderborn", "UODS"),
                            "splits", f"uods_split_eval_run_{run_idx}.json")
    trF, vaF, teF, _ = _loader_uods(manifest, amp_normalize=False)
    trT, vaT, teT, _ = _loader_uods(manifest, amp_normalize=True)
    d1 = max(float(np.abs(np.asarray(a.dataset.rms_z) - np.asarray(b.dataset.rms_z)).max())
             for a, b in ((trF, trT), (vaF, vaT), (teF, teT)))
    tr_w = np.asarray(trF.dataset.windows)
    z_te = _independent_zvector(np.asarray(teF.dataset.windows), tr_w, 2048)
    d2 = float(np.abs(z_te - np.asarray(teF.dataset.rms_z)).max())
    edges = getattr(teF.dataset, "amp_band_edges", None)
    return {"data": "UODS", "fold": f"eval_run_{run_idx}", "edges": edges,
            "max|Δ|_ampnorm_invariance": d1, "max|Δ|_independent_rederive": d2,
            "n_te": int(len(teF.dataset.windows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pu_folds", nargs="+", default=["012to3_LONO1", "123to0_LONO4"])
    ap.add_argument("--uods_runs", type=int, nargs="+", default=[5, 100])
    args = ap.parse_args()
    rows = []
    for f in args.pu_folds:
        split, lono = f.rsplit("_LONO", 1)
        rows.append(verify_pu(split, int(lono)))
    for r in args.uods_runs:
        rows.append(verify_uods(r))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
