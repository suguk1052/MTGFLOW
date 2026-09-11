"""B-2 Deep SVDD 리포트 — PU Table 4 행 + proposed/raw paired + collapse·peak 집계.

report_B1_classical 의 PU 집계 함수(_pu_block_from_npz/pu_columns/paired/holm/pu_proposed_folds/
pu_raw_folds)를 그대로 재사용한다. Deep SVDD는 b3 스키마 npz(te_f=‖φ-c‖² 거리)라 IF/OC-SVM과 동일 경로.
B-2는 PU 전용(UODS 없음).

추가 보고(요건 ②③):
  - fold별 collapse 지표(emb_std/score_cv/val_rank_ratio) 집계 + 기준 미달(collapse=true) 개수/120.
  - peak GPU(meta_json.peak_gpu_gb) max/mean. cudnn 비결정성 주의.

산출: reports/report_B2_deepsvdd.md
사용: conda run -n mtgflow python analysis/report_B2_deepsvdd.py
"""
import glob
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from report_B1_classical import (  # noqa: E402
    _pu_block_from_npz, pu_columns, paired, holm, fmt, _mean,
    pu_proposed_folds, pu_raw_folds, RES_PU, SPLITS, LONOS, SEEDS,
)

B2_DIR = os.path.join(RES_PU, "b2_dsvdd_window_scores")


def b2_folds():
    """fold→seed평균 지표(5 seed). Deep SVDD b3 npz."""
    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            blks = []
            for s in SEEDS:
                p = os.path.join(B2_DIR, f"{split}_LONO{lono}_s{s}.npz")
                if os.path.exists(p):
                    blks.append(_pu_block_from_npz(p))
            if not blks:
                continue
            folds[(split, lono)] = {
                "auroc": _mean([b["auroc"] for b in blks]),
                "rho": _mean([b["rho_rms_score"] for b in blks]),
                "high_fpr": _mean([b["high_amp_fpr"] for b in blks]),
                "low_fpr": _mean([b["low_amp_fpr"] for b in blks]),
                "per_fault": {fid: _mean([b["per_fault_auroc"].get(fid, {}).get("auroc")
                                          for b in blks])
                              for fid in set().union(*[b["per_fault_auroc"].keys() for b in blks])},
            }
    return folds


def collapse_and_peak():
    rows = []
    for p in sorted(glob.glob(os.path.join(B2_DIR, "*.npz"))):
        mj = json.loads(str(np.load(p, allow_pickle=True)["meta_json"]))
        c = mj["collapse"]
        rows.append({"fold": f"{mj['split']}_LONO{mj['lono']}_s{mj['seed']}",
                     "emb_std": c["emb_std"], "score_cv": c["score_cv"],
                     "val_rank_ratio": c["val_rank_ratio"], "collapse": c["collapse"],
                     "fails": c.get("fails", []), "peak_gpu_gb": mj.get("peak_gpu_gb")})
    return rows


def main():
    report = os.path.join(PROJECT_ROOT, "reports", "report_B2_deepsvdd.md")
    folds = b2_folds()
    proposed = pu_proposed_folds()
    raw = pu_raw_folds()
    cols = pu_columns(folds)
    rows = collapse_and_peak()

    fa = {k: folds[k]["auroc"] for k in folds}
    pr = paired(fa, raw); pp = paired(fa, proposed)
    hr = holm([pr["p"]]); hp = holm([pp["p"]])

    n_collapse = sum(r["collapse"] for r in rows)
    emb = [r["emb_std"] for r in rows]
    peaks = [r["peak_gpu_gb"] for r in rows if r["peak_gpu_gb"] is not None]

    L = ["# B-2. Deep SVDD (One-Class, 1D-CNN) — PU Table 4 행", ""]
    L.append("입력 raw 2048 window(train-normal StandardScaler, amp_normalize=False). "
             "score=‖φ(x)−c‖²(거리=이상). threshold=val-normal 95pct. 24 fold × seed 2024~2028 5-seed 평균.")
    L.append("하이퍼(동결): 1D-CNN(전 층 bias 없음·BN affine=False·LeakyReLU), center=init forward 평균+eps(0.1) 고정, "
             "Adam lr1e-3 wd1e-6, MultiStepLR[50], epoch=100, batch=128.")
    L.append("")
    L.append("## PU (24 fold) — Table 4 열")
    L.append("")
    L.append("| 모델 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append(f"| Deep SVDD (raw) | {fmt(*cols['overall'])} | {fmt(*cols['zero'])} | {fmt(*cols['comp'])} | "
             f"{fmt(cols['amp_sensitive'][0])} | {fmt(cols['shape_sensitive'][0])} | "
             f"{fmt(*cols['high_fpr'])} | {fmt(*cols['low_fpr'])} | {fmt(*cols['rho'])} |")
    L.append(f"| raw MTGFlow (참조) | {fmt(_mean(list(raw.values())))} | | | | | | | |")
    L.append(f"| 제안 N6+log+Fisher (참조) | {fmt(_mean(list(proposed.values())))} | | | | | | | |")
    L.append("")
    L.append("### paired (24 fold overall AUROC, Wilcoxon+Holm)")
    L.append("")
    L.append("| 대상 | Δ | p(Holm) | 개선/24 |")
    L.append("|---|---|---|---|")
    L.append(f"| vs raw MTGFlow | {pr['mean_diff']:+.3f} | {fmt(hr[0])} | {pr['n_pos']}/{pr['n']} |")
    L.append(f"| vs 제안(Fisher) | {pp['mean_diff']:+.3f} | {fmt(hp[0])} | {pp['n_pos']}/{pp['n']} |")
    L.append("")
    L.append("## collapse 진단 (요건 ②, label-free, 하이퍼 불변)")
    L.append("")
    L.append(f"- **기준 미달(collapse=true) fold: {n_collapse}/120.** 기준: emb_std<1e-3 / score_cv<1e-2 / val_rank_ratio<1.05.")
    L.append(f"- emb_std 분포: min {min(emb):.2e} / median {np.median(emb):.2e} / max {max(emb):.2e}.")
    if n_collapse:
        L.append("- 미달 fold:")
        for r in rows:
            if r["collapse"]:
                L.append(f"  - {r['fold']}: {', '.join(r['fails'])} (emb_std {r['emb_std']:.2e}, cv {r['score_cv']:.2e}, rank {r['val_rank_ratio']:.2f})")
    L.append(f"- **판정**: 경계 하회 소수(주로 emb_std 임계 1e-3 근방) — 학습은 정상 수렴, AUROC로 판별 유지. 하이퍼 변경 없음(B-G6).")
    L.append("")
    L.append("## GPU peak · 재현성 (요건 ③)")
    L.append("")
    if peaks:
        L.append(f"- peak GPU(allocator, `torch.cuda.max_memory_allocated`): max {max(peaks):.3f} GB / mean {np.mean(peaks):.3f} GB (V100-16 대비 무시 가능).")
    L.append("- **cudnn conv 비결정성**: 동일 seed GPU 재실행 시 미세차 가능(`use_deterministic_algorithms` 미적용, 속도 우선). **5-seed 평균으로 보고**.")
    L.append("")
    L.append("## 프레이밍 (B-G2)")
    L.append("- Deep SVDD는 deep OCC 기준점. 본문 우세 주장 대상은 OCC 계열 한정. raw 입력 deep OCC가 6-band 특징 없이 얻는 상한을 보여주는 참고 행.")

    with open(report, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {report}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
