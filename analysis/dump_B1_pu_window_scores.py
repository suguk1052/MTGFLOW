"""B-1 고전 baseline — PU per-window score dump (IF·OC-SVM × raw·band, CPU, 학습 없음).

제안 모델 amp 분기 target과 동일한 (b) 6-log-band z-vector를 얻기 위해, 로더를
amp_normalize=False, amp_n_bands=6, amp_band_scheme='log' 로 **직접 호출**한다(main.py 경유 금지):
  (a) raw 2048 window   = loader.dataset.windows  (N,2048)  StandardScaler 적용·amp_normalize=False라 진폭 보존
  (b) 6-log-band z-vec  = loader.dataset.rms_z    (N,6)     band 경계 [1,3,10,32,102,323,1025], train-normal z-score

로더 인자(split/ids/window/stride/sensor_mode)는 B3 raw checkpoint의 paderborn_metadata에서 복원
(b3_raw_window_scores 캐시와 동일 window 재현). amp 관련 3개(amp_normalize/amp_n_bands/amp_band_scheme)만
B-1용으로 오버라이드한다.

  ▶ amp_normalize=False 에서도 rms_z 는 학습(amp_normalize=True) target과 bit-identical:
    paderborn.py:540-545 가 band target(compute_band_rms(w[:,0]))을 amp_normalize 나눗셈 **이전**에 계산,
    z-score(667-685)는 amp_normalize 무관 train-normal 통계만 사용. (verify_B1_band_target.py 로 실증.)

산출: results/Paderborn/b1_{if,ocsvm}_{raw,band}_window_scores/<split>_LONO<n>_s<seed>.npz
  (b3_raw_window_scores 스키마 동형) te_f, va_f, tr_f, te_lab, te_ids, te_rms, va_ids, va_rms,
  tr_ids, tr_rms, meta_json. IF=seed 2024~2028 5개, OC-SVM=결정론 1개(s2024).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from diagnose_G3a_amp_bands import (  # noqa: E402
    DATA_ROOT, RESULTS_ROOT, SPLIT_INFO, LONO_TARGET, HIGH_AMP, pull,
)
from Dataset.paderborn import loader_Paderborn_OCC  # noqa: E402
from baselines.if_ocsvm import (  # noqa: E402
    fit_score_if, fit_score_ocsvm, val_threshold_and_fpr, OCSVM_SUBSAMPLE_N,
)

B1_N_BANDS = 6
B1_BAND_SCHEME = "log"
B1_SEEDS = [2024, 2025, 2026, 2027, 2028]
B3_BATCH_DIR = "LONO_B3_5seeds"
B3_PREFIX = "raw_vib"


def load_fold_meta(split, lono):
    """B3 raw checkpoint(s2024)에서 로더 복원용 metadata를 읽는다(데이터는 seed 무관)."""
    run = f"{B3_PREFIX}_{split}_LONO{lono}_s2024"
    ckpt_path = os.path.join(RESULTS_ROOT, B3_BATCH_DIR, run, "model.pth")
    if not os.path.exists(ckpt_path):
        alt = os.path.join(RESULTS_ROOT, run, "model.pth")
        ckpt_path = alt if os.path.exists(alt) else ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"B3 checkpoint 없음(로더 인자 복원 불가): {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["paderborn_metadata"]


def build_b1_loader(meta, batch_size=256):
    """B3 meta로 로더 복원 + amp 오버라이드(amp_normalize=False, 6-log-band). stdout 억제."""
    with contextlib.redirect_stdout(io.StringIO()):
        return loader_Paderborn_OCC(
            root=DATA_ROOT,
            train_loads=meta["train_load_setting"],
            test_loads=meta["test_load_setting"],
            train_ids=meta["train_ids"],
            val_ids=meta["val_ids"],
            test_norm_ids=meta["test_norm_ids"],
            exclude_ids=meta.get("exclude_ids", []),
            window_size=int(meta["window_size"]),
            stride_size=int(meta["stride_size"]),
            sensor_mode=meta["sensor_mode"],
            meta_source=meta.get("meta_source", "static"),
            measured_meta_stats=meta.get("measured_meta_stats", "meanstd"),
            amp_normalize=False,           # ▶ 오버라이드: (a) raw 진폭 보존 / rms_z는 불변
            amp_n_bands=B1_N_BANDS,        # ▶ 오버라이드: 6-band target 산출
            amp_band_scheme=B1_BAND_SCHEME,  # ▶ 오버라이드: log 경계
            amp_band_min_width=int(meta.get("amp_band_min_width", 4)),
            rms_eps=float(meta.get("rms_eps", 1e-8)),
            batch_size=batch_size,
        )


def _meta_json(split, lono, seed, model, inp, band_edges, extra):
    tgt_bid = LONO_TARGET.get(lono)
    d = {
        "split": split, "lono": int(lono), "seed": int(seed),
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_amp_group": "high" if (tgt_bid in HIGH_AMP) else "low",
        "run_name": f"b1_{model}_{inp}_{split}_LONO{lono}_s{seed}",
        "b1_model": model, "b1_input": inp,
        "amp_n_bands": B1_N_BANDS, "amp_band_scheme": B1_BAND_SCHEME,
        "amp_band_edges": (list(band_edges) if band_edges is not None else None),
    }
    d.update(extra)
    return d


def _save(out_root, model, inp, split, lono, seed, scores, common, meta_json):
    out_dir = os.path.join(out_root, f"b1_{model}_{inp}_window_scores")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_LONO{lono}_s{seed}.npz")
    np.savez_compressed(
        out_path,
        te_f=scores["te"], va_f=scores["va"], tr_f=scores["tr"],
        meta_json=json.dumps(meta_json, ensure_ascii=False), **common,
    )
    return out_path


def dump_fold(split, lono, out_root, batch_size, overwrite, seeds):
    t0 = time.time()
    # 재실행 스킵: 4 combo × (IF 5seed + OCSVM 1) 전부 존재하면 스킵
    def _exists(model, inp, seed):
        return os.path.exists(os.path.join(
            out_root, f"b1_{model}_{inp}_window_scores", f"{split}_LONO{lono}_s{seed}.npz"))
    all_done = (all(_exists("if", inp, s) for inp in ("raw", "band") for s in seeds)
                and all(_exists("ocsvm", inp, 2024) for inp in ("raw", "band")))
    if all_done and not overwrite:
        print(f"[skip] {split} LONO{lono} 전 combo 존재")
        return

    meta = load_fold_meta(split, lono)
    tr, va, te, _ = build_b1_loader(meta, batch_size)
    t_load = time.time() - t0

    tr_w, tr_rms, tr_z, tr_ids, _ = pull(tr.dataset)
    va_w, va_rms, va_z, va_ids, _ = pull(va.dataset)
    te_w, te_rms, te_z, te_ids, te_lab = pull(te.dataset)
    band_edges = getattr(te.dataset, "amp_band_edges", None)
    assert band_edges == [1, 3, 10, 32, 102, 323, 1025], f"band edges 예상과 다름: {band_edges}"

    common = dict(te_lab=te_lab, te_ids=te_ids, te_rms=te_rms,
                  va_ids=va_ids, va_rms=va_rms, tr_ids=tr_ids, tr_rms=tr_rms)
    inputs = {"raw": (tr_w, va_w, te_w), "band": (tr_z, va_z, te_z)}
    log = {"split": split, "lono": lono, "t_load": round(t_load, 1),
           "n_tr": int(len(tr_w)), "n_va": int(len(va_w)), "n_te": int(len(te_w)),
           "combos": {}}

    for inp, (Xtr, Xva, Xte) in inputs.items():
        # --- IF: 5 seed ---
        for seed in seeds:
            ts = time.time()
            sc = fit_score_if(Xtr, Xva, Xte, seed=seed)
            thr, vfpr = val_threshold_and_fpr(sc["va"])
            mj = _meta_json(split, lono, seed, "if", inp, band_edges,
                            {"n_estimators": 100, "n_fit": sc["n_fit"],
                             "val_thr95": thr, "val_fpr": vfpr})
            _save(out_root, "if", inp, split, lono, seed, sc, common, mj)
            log["combos"][f"if_{inp}_s{seed}"] = {
                "t": round(time.time() - ts, 1), "val_fpr": round(vfpr, 4)}
        # --- OC-SVM: 결정론 1회(s2024) ---
        cap = OCSVM_SUBSAMPLE_N if inp == "raw" else None
        ts = time.time()
        sc = fit_score_ocsvm(Xtr, Xva, Xte, subsample_cap=cap)
        thr, vfpr = val_threshold_and_fpr(sc["va"])
        mj = _meta_json(split, lono, 2024, "ocsvm", inp, band_edges,
                        {"nu": 0.05, "gamma": "scale", "n_fit": sc["n_fit"],
                         "n_support": sc["n_support"], "subsample_cap": cap,
                         "val_thr95": thr, "val_fpr": vfpr})
        _save(out_root, "ocsvm", inp, split, lono, 2024, sc, common, mj)
        log["combos"][f"ocsvm_{inp}"] = {
            "t": round(time.time() - ts, 1), "val_fpr": round(vfpr, 4),
            "n_fit": sc["n_fit"], "n_sv": sc["n_support"]}

    log["t_total"] = round(time.time() - t0, 1)
    print(json.dumps(log, ensure_ascii=False))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT_INFO.keys()), help="한 fold 처리(잡 단위). 미지정시 전체.")
    ap.add_argument("--lono", type=int, help="1..6. --split과 함께 한 fold 처리.")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=B1_SEEDS)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_root", type=str, default=RESULTS_ROOT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.split and args.lono:
        folds = [(args.split, args.lono)]
    else:
        folds = [(s, l) for s in args.splits for l in args.lonos]
    print(f"B-1 PU dump: {len(folds)} fold, seeds={args.seeds}, out_root={args.out_root}")
    for split, lono in folds:
        dump_fold(split, lono, args.out_root, args.batch_size, args.overwrite, args.seeds)


if __name__ == "__main__":
    main()
