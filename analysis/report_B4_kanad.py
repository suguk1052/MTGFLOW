"""B-4 KAN-AD 리포트 — PU Table 4 행 + proposed/raw paired (report_B2_deepsvdd 미러).

b4_kanad npz(b3 스키마, te_f=재구성오차 score)를 report_B1_classical의 PU 집계 함수로 처리.
KAN-AD = 재구성형 TSAD 참고 행(B-G2: 시점탐지 설계·K=1·seq2048). PU 전용(UODS 없음).

산출: reports/report_B4_kanad.md
사용: conda run -n mtgflow python analysis/report_B4_kanad.py
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

B4_DIR = os.path.join(RES_PU, "b4_kanad_window_scores")


def b4_folds():
    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            blks = []
            for s in SEEDS:
                p = os.path.join(B4_DIR, f"{split}_LONO{lono}_s{s}.npz")
                if os.path.exists(p):
                    blks.append(_pu_block_from_npz(p))
            if not blks:
                continue
            folds[(split, lono)] = {
                "auroc": _mean([b["auroc"] for b in blks]),
                "rho": _mean([b["rho_rms_score"] for b in blks]),
                "high_fpr": _mean([b["high_amp_fpr"] for b in blks]),
                "low_fpr": _mean([b["low_amp_fpr"] for b in blks]),
                "per_fault": {fid: _mean([b["per_fault_auroc"].get(fid, {}).get("auroc") for b in blks])
                              for fid in set().union(*[b["per_fault_auroc"].keys() for b in blks])},
            }
    return folds


def peak_and_pilot():
    peaks, pilot = [], None
    for p in glob.glob(os.path.join(B4_DIR, "*.npz")):
        mj = json.loads(str(np.load(p, allow_pickle=True)["meta_json"]))
        if mj.get("peak_gpu_gb") is not None:
            peaks.append(mj["peak_gpu_gb"])
        if mj["split"] == "012to3" and mj["lono"] == 1 and mj["seed"] == 2024:
            from sklearn.metrics import roc_auc_score
            z = np.load(p, allow_pickle=True)
            pilot = float(roc_auc_score(z["te_lab"], z["te_f"]))
    return peaks, pilot


def main():
    report = os.path.join(PROJECT_ROOT, "reports", "report_B4_kanad.md")
    folds = b4_folds(); proposed = pu_proposed_folds(); raw = pu_raw_folds()
    cols = pu_columns(folds)
    peaks, pilot = peak_and_pilot()
    fa = {k: folds[k]["auroc"] for k in folds}
    pr = paired(fa, raw); pp = paired(fa, proposed)
    hr = holm([pr["p"]]); hp = holm([pp["p"]])

    L = ["# B-4. KAN-AD (재구성형 TSAD, 참고 행) — PU Table 4 행", ""]
    L.append("입력 raw 2048 window(train-normal StandardScaler, amp_normalize=False). score=mean_L(recon−x)². "
             "threshold=val-normal 95pct. 24 fold × seed 2024~2028 5-seed 평균.")
    L.append("어댑터: TSLib exp 래퍼 미사용(train-normal 학습·val-normal loss checkpoint·anomaly_ratio/PA 미사용). "
             "하이퍼(동결): order(d_model)=4, lr0.01, batch128, epoch100, Adam, MSE 재구성.")
    L.append("")
    L.append("## PU (24 fold) — Table 4 열")
    L.append("")
    L.append("| 모델 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append(f"| KAN-AD (raw, 참고) | {fmt(*cols['overall'])} | {fmt(*cols['zero'])} | {fmt(*cols['comp'])} | "
             f"{fmt(cols['amp_sensitive'][0])} | {fmt(cols['shape_sensitive'][0])} | "
             f"{fmt(*cols['high_fpr'])} | {fmt(*cols['low_fpr'])} | {fmt(*cols['rho'])} |")
    L.append(f"| raw MTGFlow (참조) | {fmt(_mean(list(raw.values())))} | | | | | | | |")
    L.append(f"| 제안 N6+log+Fisher (참조) | {fmt(_mean(list(proposed.values())))} | | | | | | | |")
    L.append("")
    L.append("### paired (24 fold overall AUROC, Wilcoxon+Holm)")
    L.append("| 대상 | Δ(KAN-AD−대상) | p(Holm) | KAN-AD 우세/24 |")
    L.append("|---|---|---|---|")
    L.append(f"| vs raw MTGFlow | {pr['mean_diff']:+.3f} | {fmt(hr[0])} | {pr['n_pos']}/{pr['n']} |")
    L.append(f"| vs 제안(Fisher) | {pp['mean_diff']:+.3f} | {fmt(hp[0])} | {pp['n_pos']}/{pp['n']} |")
    L.append("")
    L.append("## 실측·주의")
    if peaks:
        L.append(f"- peak GPU(`torch.cuda.max_memory_allocated`): max {max(peaks):.3f} GB / mean {np.mean(peaks):.3f} GB. n_params 4.20M.")
    L.append(f"- **파일럿 비대표성**: 파일럿 단일 fold(012to3_LONO1_s2024)={fmt(pilot)}는 고진폭 K001 hard fold. "
             f"24 fold 평균 {fmt(cols['overall'][0])}로 훨씬 높음 — 파일럿 AUROC로 하이퍼 미변경(B-G6).")
    L.append("- **cudnn conv 비결정성**: 동일 seed GPU 재실행 미세차 가능(`use_deterministic_algorithms` 미적용) → 5-seed 평균 보고.")
    L.append("- **프레이밍(B-G2)**: KAN-AD는 최신 TSAD 참고 행. ① 시점 단위 탐지용 설계 ② K=1 채널이라 채널 융합 비활성 "
             "③ seq_len 2048 제약. '우세' 주장 대상 아님(우세 주장은 OCC 계열 한정).")
    with open(report, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {report}\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
