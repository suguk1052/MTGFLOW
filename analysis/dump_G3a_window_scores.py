"""작업 G-5 Step 1 — G-3a per-window score 캐시 dump (forward-only 재추론, 학습 없음).

G-3a per-fold JSON에는 집계 지표만 있고 per-window S_shape/S_amp 원 배열이 없어, 추론단 fusion
규칙을 바꾸려면(G-5 Fisher-tail 등) checkpoint에서 per-window score를 다시 확보해야 한다.
이 스크립트는 diagnose_G3a_amp_bands.eval_g3a 전반부와 동일하게 g3a checkpoint를 재추론해
tr/va/te per-window (S_shape, S_amp) + 라벨·bearing id·원 RMS를 .npz로 **영구 캐시**한다.
이후 fusion 실험(G-5)·G-6 등은 이 캐시만으로 GPU 없이 재조합할 수 있다.

fusion·집계·지표는 여기서 하지 않는다(순수 dump). 지표 계산·fusion 비교는 diagnose_G5_tail_fusion.py.

산출: results/Paderborn/g3a_window_scores/<split>_LONO<n>_s<seed>.npz
  배열: te_sh, te_am, va_sh, va_am, tr_sh, tr_am,
        te_lab, te_ids, te_rms, va_ids, va_rms, tr_ids, tr_rms
  meta(스칼라/문자): split, lono, seed, fold_type, desc, target_amp_group, target_norm_ids, amp_n_bands

전제: g3a checkpoint = results/Paderborn/[<batch_dir>/]<prefix>_<split>_LONO<n>_s<seed>/model.pth
"""
import argparse
import json
import os

import numpy as np
import torch

# diagnose_G3a_amp_bands 의 검증된 로더/모델/추론 함수를 그대로 재사용(복사 금지).
from diagnose_G3a_amp_bands import (  # noqa: E402
    LONO_TARGET,
    HIGH_AMP,
    RESULTS_ROOT,
    SPLIT_INFO,
    branch_scores,
    build_loader,
    build_model,
    pull,
)

OUT_DIR = os.path.join(RESULTS_ROOT, "g3a_window_scores")


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
    # eval_g3a 와 동일한 위생 검증(shape-only window + amp branch + no-meta 전용 checkpoint).
    assert bool(meta.get("amp_branch", False)), f"amp_branch=False: {run_name}"
    assert bool(meta.get("amp_normalize", False)), f"amp_normalize=False: {run_name}"
    assert not bool(meta["use_meta"]), f"use_meta=True: {run_name}"
    n_bands = int(meta.get("amp_n_bands", 1))
    assert n_bands >= 1, f"amp_n_bands={n_bands}: {run_name}"

    train_loader, val_loader, test_loader, n_sensor = build_loader(meta, batch_size)
    model = build_model(ckpt, n_sensor, device, amp_branch=True, amp_n_bands=n_bands)

    tr_w, tr_rms, tr_z, tr_ids, _ = pull(train_loader.dataset)
    va_w, va_rms, va_z, va_ids, _ = pull(val_loader.dataset)
    te_w, te_rms, te_z, te_ids, te_lab = pull(test_loader.dataset)

    tr_sh, tr_am = branch_scores(model, tr_w, tr_z, device, n_bands, batch_size)
    va_sh, va_am = branch_scores(model, va_w, va_z, device, n_bands, batch_size)
    te_sh, te_am = branch_scores(model, te_w, te_z, device, n_bands, batch_size)

    tgt = te_lab == 0
    tgt_bid = LONO_TARGET.get(lono)
    meta_json = {
        "split": split, "lono": int(lono), "seed": int(seed),
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_amp_group": "high" if (tgt_bid in HIGH_AMP) else "low",
        "target_norm_ids": sorted(set(te_ids[tgt].tolist())),
        "amp_n_bands": n_bands, "run_name": run_name,
    }

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_path,
        te_sh=te_sh, te_am=te_am, va_sh=va_sh, va_am=va_am, tr_sh=tr_sh, tr_am=tr_am,
        te_lab=te_lab, te_ids=te_ids, te_rms=te_rms,
        va_ids=va_ids, va_rms=va_rms, tr_ids=tr_ids, tr_rms=tr_rms,
        meta_json=json.dumps(meta_json, ensure_ascii=False),
    )
    print(f"    wrote {out_path}  (te {len(te_sh)} / va {len(va_sh)} / tr {len(tr_sh)} window, {n_bands}밴드)")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--g3a_batch_dir", type=str, default="")
    ap.add_argument("--g3a_prefix", type=str, default="g3a")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()), choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 .npz도 다시 dump")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  seed={args.seed}  out_dir={args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    n_ok = n_skip = n_miss = 0
    for split in args.splits:
        for lono in args.lonos:
            print(f"[{split} LONO{lono} s{args.seed}]")
            r = dump_fold(args.g3a_batch_dir, args.g3a_prefix, split, lono, args.seed,
                          device, args.batch_size, args.out_dir, args.overwrite)
            n_ok += (r == "ok"); n_skip += (r == "skip"); n_miss += (r is None)

    print(f"\n=== dump 완료: ok={n_ok} skip={n_skip} miss={n_miss} (seed={args.seed}) ===")


if __name__ == "__main__":
    main()
