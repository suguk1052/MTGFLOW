"""작업 G-7 Step 0 — raw(B3) baseline per-window score 캐시 dump (forward-only 재추론, 학습 없음).

G-3a per-window 캐시(g3a_window_scores/*.npz)에는 shape/amp branch score만 있고 raw baseline
(B3 flow NLL, amp_normalize=False 별개 checkpoint)의 per-window 원배열이 없다. G-7 진단의
표 2(고정 임계값 per-fault recall)·표 3(percentile sweep)은 raw per-window가 있어야 계산되므로,
이 스크립트가 B3 checkpoint를 eval_raw 와 동일하게 forward-only 재추론해 tr/va/te raw NLL을
.npz로 **영구 캐시**한다. 이후 diagnose_G7 은 이 캐시 + g3a 캐시만으로 GPU 없이 완결된다.

  raw = flow_nll(model, x) = -model.test(x) = -log p_x  (클수록 이상)
  checkpoint = results/Paderborn/LONO_B3_5seeds/raw_vib_<split>_LONO<n>_s<seed>/model.pth
              (amp_branch=False, amp_normalize=False, use_meta=False)

⚠️ te·va 뿐 아니라 **tr(train)도 dump**한다. diagnose_G7 의 진폭군 정상 FPR은 정상 pool
   (train+val+test-normal) per-bearing 평균이라(=_score_block 규약, diag_G3a 와 동일), train을
   빼면 published 저진폭 FPR(0.418 등)을 재현할 수 없다. 정합성 우선.

산출: results/Paderborn/b3_raw_window_scores/<split>_LONO<n>_s<seed>.npz
  배열: te_f, va_f, tr_f, te_lab, te_ids, te_rms, va_ids, va_rms, tr_ids, tr_rms
  meta(문자): split, lono, seed, fold_type, desc, target_amp_group, target_norm_ids, run_name
"""
import argparse
import json
import os

import numpy as np
import torch

# diagnose_G3a_amp_bands 의 검증된 로더/모델/추론 함수를 그대로 재사용(복사 금지).
from diagnose_G3a_amp_bands import (  # noqa: E402
    HIGH_AMP,
    LONO_TARGET,
    RESULTS_ROOT,
    SPLIT_INFO,
    build_loader,
    build_model,
    flow_nll,
    pull,
)

OUT_DIR = os.path.join(RESULTS_ROOT, "b3_raw_window_scores")


def _ckpt_path(batch_dir, prefix, split, lono, seed):
    run_name = f"{prefix}_{split}_LONO{lono}_s{seed}"
    p = os.path.join(RESULTS_ROOT, batch_dir, run_name, "model.pth")
    if not os.path.exists(p):
        alt = os.path.join(RESULTS_ROOT, run_name, "model.pth")
        p = alt if os.path.exists(alt) else p
    return run_name, p


def dump_fold(batch_dir, prefix, split, lono, seed, device, batch_size, out_dir, overwrite):
    out_path = os.path.join(out_dir, f"{split}_LONO{lono}_s{seed}.npz")
    if os.path.exists(out_path) and not overwrite:
        print(f"    [skip] 이미 존재: {out_path}")
        return "skip"

    run_name, ckpt_path = _ckpt_path(batch_dir, prefix, split, lono, seed)
    if not os.path.exists(ckpt_path):
        print(f"    [skip] checkpoint 없음: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    # raw baseline 위생 검증: amp branch 없음 + shape window 아님 + no-meta.
    assert not bool(meta.get("amp_branch", False)), f"amp_branch=True (raw 아님): {run_name}"
    assert not bool(meta.get("amp_normalize", False)), f"amp_normalize=True (raw 아님): {run_name}"
    assert not bool(meta["use_meta"]), f"use_meta=True: {run_name}"

    train_loader, val_loader, test_loader, n_sensor = build_loader(meta, batch_size)
    model = build_model(ckpt, n_sensor, device, amp_branch=False)

    tr_w, tr_rms, _, tr_ids, _ = pull(train_loader.dataset)
    va_w, va_rms, _, va_ids, _ = pull(val_loader.dataset)
    te_w, te_rms, _, te_ids, te_lab = pull(test_loader.dataset)

    tr_f = flow_nll(model, tr_w, device, batch_size)
    va_f = flow_nll(model, va_w, device, batch_size)
    te_f = flow_nll(model, te_w, device, batch_size)

    tgt = te_lab == 0
    tgt_bid = LONO_TARGET.get(lono)
    meta_json = {
        "split": split, "lono": int(lono), "seed": int(seed),
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_amp_group": "high" if (tgt_bid in HIGH_AMP) else "low",
        "target_norm_ids": sorted(set(te_ids[tgt].tolist())),
        "run_name": run_name,
    }

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_path,
        te_f=te_f, va_f=va_f, tr_f=tr_f,
        te_lab=te_lab, te_ids=te_ids, te_rms=te_rms,
        va_ids=va_ids, va_rms=va_rms, tr_ids=tr_ids, tr_rms=tr_rms,
        meta_json=json.dumps(meta_json, ensure_ascii=False),
    )
    print(f"    wrote {out_path}  (te {len(te_f)} / va {len(va_f)} / tr {len(tr_f)} window)")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028],
                    help="복수 seed 일괄 dump (기본 5-seed).")
    ap.add_argument("--raw_batch_dir", type=str, default="LONO_B3_5seeds")
    ap.add_argument("--raw_prefix", type=str, default="raw_vib")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()), choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 .npz도 다시 dump")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  seeds={args.seeds}  out_dir={args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    n_ok = n_skip = n_miss = 0
    for seed in args.seeds:
        for split in args.splits:
            for lono in args.lonos:
                print(f"[{split} LONO{lono} s{seed}]")
                r = dump_fold(args.raw_batch_dir, args.raw_prefix, split, lono, seed,
                              device, args.batch_size, args.out_dir, args.overwrite)
                n_ok += (r == "ok"); n_skip += (r == "skip"); n_miss += (r is None)

    print(f"\n=== raw dump 완료: ok={n_ok} skip={n_skip} miss={n_miss} ===")


if __name__ == "__main__":
    main()
