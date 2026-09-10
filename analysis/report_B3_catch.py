"""B-3 CATCH 표 편입 (재사용 검증, analysis-only, 학습·재추론 없음).

B-0에서 CATCH checkpoint 규칙 = val-normal 재구성 loss 선택 확인(누수 없음, 재추론 불필요).
여기서는 `../CATCH/result_pu/LOSO_<split>_LONO<n>_s<seed>/metrics.json`의 `auroc`를 24 LOSO fold ×
5 seed 평균해 **overall = LOSO 0.539**(표값)와 fold 유형별(zero-support/compositional) 값을 낸다.
proposed/raw(24 fold) 대비 paired Wilcoxon+Holm도 산출.

한계(B-G1): CATCH `scores.npz`는 point-level이고 window id/rms 미보유 → **amp/shape-sensitive·진폭군
FPR·ρ(RMS,score) 열은 산출 불가 → `—(CATCH per-window id·rms 없음)`**로 표기(행 삭제 금지).

산출: reports/report_B3_catch.md
사용: conda run -n mtgflow python analysis/report_B3_catch.py
"""
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from report_B1_classical import paired, holm, fmt, _mean, _ms, pu_proposed_folds, pu_raw_folds  # noqa: E402
from diagnose_G3a_amp_bands import SPLIT_INFO  # noqa: E402

CATCH_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "CATCH", "result_pu"))
SPLITS = ["123to0", "023to1", "013to2", "012to3"]
LONOS = [1, 2, 3, 4, 5, 6]
SEEDS = [2024, 2025, 2026, 2027, 2028]
NA = "—(CATCH per-window id·rms 없음)"


def catch_folds():
    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            vs = []
            for s in SEEDS:
                m = os.path.join(CATCH_ROOT, f"LOSO_{split}_LONO{lono}_s{s}", "metrics.json")
                if os.path.exists(m):
                    vs.append(float(json.load(open(m))["auroc"]))
            if vs:
                folds[(split, lono)] = float(np.mean(vs))
    return folds


def main():
    report = os.path.join(PROJECT_ROOT, "reports", "report_B3_catch.md")
    folds = catch_folds()
    proposed = pu_proposed_folds()
    raw = pu_raw_folds()

    keys = list(folds)
    zero = [folds[k] for k in keys if SPLIT_INFO[k[0]][1] == "zero-support"]
    comp = [folds[k] for k in keys if SPLIT_INFO[k[0]][1] == "compositional"]
    pr = paired(folds, raw); pp = paired(folds, proposed)
    hr = holm([pr["p"]]); hp = holm([pp["p"]])

    L = ["# B-3. CATCH (재구성형 TSAD, ICLR'25) — 표 편입 (재사용 검증)", ""]
    L.append("B-0에서 checkpoint 규칙 = val-normal 재구성 loss 선택 확인(누수 0) → **재추론 불필요, metrics.json 재사용**.")
    L.append(f"LOSO 24 fold × 5 seed = {sum(1 for _ in keys)*5}개 metrics.json의 `auroc` 집계. patch 64/64, seq_len 2048, K=1 채널.")
    L.append("")
    L.append("## PU (24 LOSO fold) — Table 4 열")
    L.append("")
    L.append("| 모델 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append(f"| CATCH | {fmt(*_ms([folds[k] for k in keys]))} | {fmt(*_ms(zero))} | {fmt(*_ms(comp))} | "
             f"{NA} | {NA} | {NA} | {NA} | {NA} |")
    L.append(f"| raw MTGFlow (참조) | {fmt(_mean(list(raw.values())))} | | | | | | | |")
    L.append(f"| 제안 N6+log+Fisher (참조) | {fmt(_mean(list(proposed.values())))} | | | | | | | |")
    L.append("")
    L.append("### paired (24 fold overall AUROC, Wilcoxon+Holm)")
    L.append("| 대상 | Δ(CATCH−대상) | p(Holm) | CATCH 우세/24 |")
    L.append("|---|---|---|---|")
    L.append(f"| vs raw MTGFlow | {pr['mean_diff']:+.3f} | {fmt(hr[0])} | {pr['n_pos']}/{pr['n']} |")
    L.append(f"| vs 제안(Fisher) | {pp['mean_diff']:+.3f} | {fmt(hp[0])} | {pp['n_pos']}/{pp['n']} |")
    L.append("")
    L.append("## 한계·각주 (B-G1/B-G2)")
    L.append(f"- **{NA}**: CATCH `scores.npz`는 point-level(299,521 등)이고 window별 bearing id·RMS를 저장하지 않아 "
             "subgroup(amp/shape-sensitive)·진폭군 정상 FPR·ρ(RMS,score) 열은 산출 불가. 행은 유지하고 사유 표기(삭제 금지).")
    L.append("- **프레이밍(B-G2)**: CATCH는 표준 재구성형 TSAD 설계의 참고 행. ① 시점 단위 탐지용 설계 ② K=1 채널이라 "
             "채널 간 융합 비활성 ③ seq_len 2048 제약. '우세' 주장 대상 아님.")
    L.append("- 표값 = LOSO 0.539(24셀×5seed). B2 pooled(0.524)는 표 미사용.")
    L.append("")
    L.append("> analysis-only. 재추론·모델 변경 없음.")

    with open(report, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {report}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
