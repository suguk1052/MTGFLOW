#!/usr/bin/env python3
"""P-2 파일럿 6조건 게이트 검증 (CPU, 학습 없음).

조건(사용자 지정 2026-09-07):
  1. 실제 loader 경계(ckpt metadata amp_band_edges) == EDA 재현 경계
  2. K=6 및 checkpoint metadata의 scheme·edges 일치
  3. 모든 band score(dump npz)와 loss(train_log)에 NaN/Inf 없음
  4. 학습→test→window-score dump→Fisher diag 전체 산출물 존재
  5. dump 재구성 equal-z(fusion_A) == 해당 test metrics(overall_auroc_total_std) 허용오차 내
  6. log의 3-bin·7-bin 저주파 밴드 log-RMS 분산·유효 표본 병리 없음

사용: conda run -n mtgflow python analysis/verify_P2_pilot.py --schemes energy log --split 023to1 --lono 2 --seed 2026
반환코드 0=전부 통과, 1=하나라도 실패(자동 제출 금지).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from Dataset.paderborn import compute_band_boundaries  # noqa: E402
from eda_P2_band_partition import g3a_ckpt_meta, train_normal_mean_psd, N_BANDS, MIN_WIDTH  # noqa: E402

RES = os.path.join(PROJECT_ROOT, "results", "Paderborn")
TOL_EQUALZ = 1e-3
STD_FLOOR = 1e-3


def ckpt_path(prefix, split, lono, seed):
    return os.path.join(RES, f"{prefix}_{split}_LONO{lono}_s{seed}", "model.pth")


def eda_edges_for(scheme, split, lono, seed):
    F = 2048 // 2 + 1
    if scheme == "energy":
        meta, _ = g3a_ckpt_meta(split, lono, seed)
        psd, _, F = train_normal_mean_psd(meta)
        _, info = compute_band_boundaries(F, N_BANDS, "energy", psd=psd,
                                          min_width=MIN_WIDTH, return_info=True)
        return list(info["edges"])
    _, info = compute_band_boundaries(F, N_BANDS, scheme, return_info=True)
    if info["edges"] is not None:
        return list(info["edges"])
    groups = compute_band_boundaries(F, N_BANDS, scheme)
    return [int(np.asarray(g)[0]) for g in groups] + [F]


def check_scheme(scheme, split, lono, seed):
    tag = f"{scheme}/{split}/LONO{lono}/s{seed}"
    fails = []
    prefix = f"p2_{scheme}"
    cp = ckpt_path(prefix, split, lono, seed)
    if not os.path.exists(cp):
        return [f"[{tag}] C4 실패: checkpoint 없음 {cp}"], {}
    ck = torch.load(cp, map_location="cpu")
    meta = ck.get("paderborn_metadata", {})
    n_bands = int(meta.get("amp_n_bands", -1))
    sc = meta.get("amp_band_scheme")
    edges_ck = meta.get("amp_band_edges")

    if n_bands != N_BANDS:
        fails.append(f"[{tag}] C2 실패: amp_n_bands={n_bands} != {N_BANDS}")
    if sc != scheme:
        fails.append(f"[{tag}] C2 실패: ckpt scheme={sc} != {scheme}")

    edges_eda = eda_edges_for(scheme, split, lono, seed)
    if edges_ck is None:
        fails.append(f"[{tag}] C1 실패: ckpt amp_band_edges 없음")
    elif list(edges_ck) != list(edges_eda):
        fails.append(f"[{tag}] C1 실패: ckpt edges {edges_ck} != EDA edges {edges_eda}")

    npz = os.path.join(RES, f"{prefix}_window_scores", f"{split}_LONO{lono}_s{seed}.npz")
    if not os.path.exists(npz):
        fails.append(f"[{tag}] C4 실패: dump npz 없음 {npz}")
    else:
        z = np.load(npz, allow_pickle=False)
        for k in ("tr_sh", "tr_am", "va_sh", "va_am", "te_sh", "te_am"):
            a = z[k]
            if not np.all(np.isfinite(a)):
                fails.append(f"[{tag}] C3 실패: dump {k}에 NaN/Inf ({(~np.isfinite(a)).sum()}개)")

    tl = os.path.join(RES, f"{prefix}_{split}_LONO{lono}_s{seed}", "train_log.jsonl")
    if os.path.exists(tl):
        bad = 0
        with open(tl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                for kk, vv in d.items():
                    if isinstance(vv, (int, float)) and not np.isfinite(vv):
                        bad += 1
        if bad:
            fails.append(f"[{tag}] C3 실패: train_log에 비유한 값 {bad}개")
    else:
        fails.append(f"[{tag}] C3 경고: train_log.jsonl 없음 {tl}")

    diag = os.path.join(RES, f"diag_P2_{scheme}_tail_fusion", f"{split}_LONO{lono}_s{seed}.json")
    if not os.path.exists(diag):
        fails.append(f"[{tag}] C4 실패: Fisher diag JSON 없음 {diag}")
    else:
        dj = json.load(open(diag))
        fa = dj.get("auroc", {}).get("fusion_A")
        fb = dj.get("auroc", {}).get("fusion_B")
        if fb is None:
            fails.append(f"[{tag}] C4 실패: diag fusion_B(Fisher) 없음")
        tm = os.path.join(RES, f"{prefix}_{split}_LONO{lono}_s{seed}", "paderborn_per_bearing_metrics.json")
        tjson = json.load(open(tm))
        total_std = tjson.get("overall_auroc_total_std")
        if fa is None or total_std is None:
            fails.append(f"[{tag}] C5 실패: fusion_A={fa} 또는 total_std={total_std} 없음")
        elif abs(fa - total_std) > TOL_EQUALZ:
            fails.append(f"[{tag}] C5 실패: |equal_z {fa:.6f} - test total_std {total_std:.6f}|={abs(fa-total_std):.2e} > {TOL_EQUALZ}")
        else:
            print(f"  [C5] {tag}: equal_z {fa:.6f} ~= test total_std {total_std:.6f} (Δ={abs(fa-total_std):.2e}) OK")

    if scheme == "log":
        std = meta.get("train_logrms_std")
        mean = meta.get("train_logrms_mean")
        if not isinstance(std, list) or len(std) != N_BANDS:
            fails.append(f"[{tag}] C6 실패: train_logrms_std 형식 이상 {std}")
        else:
            for bi in (0, 1):
                sname = {0: "band0(3bin)", 1: "band1(7bin)"}[bi]
                sv = float(std[bi])
                if not np.isfinite(sv) or sv < STD_FLOOR:
                    fails.append(f"[{tag}] C6 실패: {sname} log-RMS std={sv:.2e} < {STD_FLOOR}(병리)")
                if not np.isfinite(float(mean[bi])):
                    fails.append(f"[{tag}] C6 실패: {sname} log-RMS mean 비유한")
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=False)
            n_tr = len(z["tr_sh"])
            if n_tr < 100:
                fails.append(f"[{tag}] C6 실패: train window 수 {n_tr} 과소")

    info = {"scheme": scheme, "n_bands": n_bands, "ckpt_scheme": sc,
            "edges_ck": edges_ck, "edges_eda": edges_eda,
            "train_logrms_std": meta.get("train_logrms_std")}
    return fails, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["energy", "log"])
    ap.add_argument("--split", default="023to1")
    ap.add_argument("--lono", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    all_fails = []
    for sc in args.schemes:
        fails, info = check_scheme(sc, args.split, args.lono, args.seed)
        print(f"\n===== {sc} ({args.split}/LONO{args.lono}/s{args.seed}) =====")
        print(f"  n_bands={info.get('n_bands')} scheme={info.get('ckpt_scheme')}")
        print(f"  edges_ck ={info.get('edges_ck')}")
        print(f"  edges_eda={info.get('edges_eda')}")
        if info.get("train_logrms_std") is not None:
            print(f"  train_logrms_std={[round(float(x),4) for x in info['train_logrms_std']]}")
        if fails:
            for f in fails:
                print("  ❌ " + f)
        else:
            print("  ✅ 6조건 통과")
        all_fails += fails

    print("\n" + "=" * 60)
    if all_fails:
        print(f"❌ 게이트 실패: {len(all_fails)}건 -> 전량 제출 금지")
        sys.exit(1)
    print("✅ 게이트 통과(전 scheme 6조건) -> 전량 제출 가능")
    sys.exit(0)


if __name__ == "__main__":
    main()
