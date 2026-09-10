"""B-5 진단 (analysis-only, CPU, 학습 없음) — 같은 6-band 특징 위 밀도/OCC 모델 비교.

B-1/B-2 결과("이득=log-band 특징")를 정밀 분해한다. 제안 모델의 amp 분기(S_amp=조건부 Gaussian
head on 6-band)가 **동일 6-band 특징**을 쓰는 generic OCC(OC-SVM·band / IF·band) 대비 순이득이
있는지 24 fold paired(PU) + 100 split paired(UODS)로 검정. 모델·하이퍼 변경 없음(기존 캐시 재사용).

재사용 캐시:
  제안 per-window  results/Paderborn/p2_log_window_scores/*.npz (te_sh/te_am/va_*/tr_*)
                   results/UODS/uods_window_scores/uods_eval_run*_proposed_s*.npz
  고전 band        results/{Paderborn,UODS}/b1_{if,ocsvm}_band_window_scores/*.npz (B-1)
  Fisher = fisher_tail(te_sh,te_am,va_sh,va_am).

산출: reports/report_B5_feature_vs_model.md
사용: conda run -n mtgflow python analysis/report_B5_feature_vs_model.py
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
    pu_combo_folds, uods_combo_splits, paired, holm, fmt, _mean, _ms,
    RES_PU, RES_UODS, SPLITS, LONOS, SEEDS,
)
from diagnose_G5_tail_fusion import fisher_tail  # noqa: E402
from diagnose_G3a_amp_bands import (  # noqa: E402
    _score_block, SPLIT_INFO, LONO_TARGET, HIGH_AMP,
)
from diagnose_uods_fusion import subgroup_aurocs, fpr_at_val95  # noqa: E402

P2_LOG = os.path.join(RES_PU, "p2_log_window_scores")
UODS_CACHE = os.path.join(RES_UODS, "uods_window_scores")
EVAL_RUNS = list(range(5, 105))


# --- 제안 PU per-fold blocks (S_amp / S_shape / Fisher), seed 평균 ---
def _block(S_te, S_va, S_tr, te_lab, te_ids, te_rms, va_ids, va_rms, tr_ids, tr_rms):
    tgt = te_lab == 0
    norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]])
    norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])
    return _score_block("x", S_te, S_va, te_lab, te_ids, te_rms, S_tr, S_va,
                        te_rms[tgt], norm_rms, norm_ids, 95.0)


def proposed_pu_folds():
    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            acc = {"fisher": [], "amp": [], "shape": []}
            for s in SEEDS:
                p = os.path.join(P2_LOG, f"{split}_LONO{lono}_s{s}.npz")
                if not os.path.exists(p):
                    continue
                z = np.load(p, allow_pickle=True)
                te_sh, te_am, va_sh, va_am, tr_sh, tr_am = (z["te_sh"], z["te_am"], z["va_sh"],
                                                            z["va_am"], z["tr_sh"], z["tr_am"])
                te_lab, te_ids, te_rms = z["te_lab"], z["te_ids"], z["te_rms"]
                va_ids, va_rms, tr_ids, tr_rms = z["va_ids"], z["va_rms"], z["tr_ids"], z["tr_rms"]
                fi_te = fisher_tail(te_sh, te_am, va_sh, va_am)
                fi_va = fisher_tail(va_sh, va_am, va_sh, va_am)
                fi_tr = fisher_tail(tr_sh, tr_am, va_sh, va_am)
                args = (te_lab, te_ids, te_rms, va_ids, va_rms, tr_ids, tr_rms)
                acc["fisher"].append(_block(fi_te, fi_va, fi_tr, *args))
                acc["amp"].append(_block(te_am, va_am, tr_am, *args))
                acc["shape"].append(_block(te_sh, va_sh, tr_sh, *args))
            if not acc["fisher"]:
                continue
            folds[(split, lono)] = {
                m: {"auroc": _mean([b["auroc"] for b in acc[m]]),
                    "low_fpr": _mean([b["low_amp_fpr"] for b in acc[m]]),
                    "high_fpr": _mean([b["high_amp_fpr"] for b in acc[m]]),
                    "rho": _mean([b["rho_rms_score"] for b in acc[m]])}
                for m in ("fisher", "amp", "shape")}
    return folds


def uods_proposed_amp_splits():
    """UODS proposed S_amp per-split overall AUROC(+fpr), seed 평균."""
    per = {}
    for r in EVAL_RUNS:
        ovs, fprs = [], []
        for s in SEEDS:
            p = os.path.join(UODS_CACHE, f"uods_eval_run{r}_proposed_s{s}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            ovs.append(subgroup_aurocs(z["te_am"], z["te_state"], z["te_fam"], z["te_isball"])["overall"])
            st = np.asarray(z["te_state"], int)
            fprs.append(fpr_at_val95(np.asarray(z["te_am"])[st == 0], z["va_am"])[0])
        if ovs:
            per[r] = {"overall": _mean(ovs), "fpr_val95": _mean(fprs)}
    return per


def main():
    report = os.path.join(PROJECT_ROOT, "reports", "report_B5_feature_vs_model.md")
    prop = proposed_pu_folds()
    ocs = pu_combo_folds("ocsvm", "band")
    iff = pu_combo_folds("if", "band")

    L = ["# B-5. 특징 vs 모델 진단 — 같은 6-log-band 특징 위 밀도/OCC 모델 비교", ""]
    L.append("analysis-only(기존 캐시 재사용, 학습·모델·하이퍼 변경 없음). S_amp=제안 amp 분기(조건부 Gaussian "
             "head on 6-band), OC-SVM·band/IF·band=동일 6-band 특징의 generic OCC. Fisher/S_shape=참고.")
    L.append("")

    # ① PU 24 fold별 표
    L.append("## ① PU 24 fold별 AUROC (seed 평균)")
    L.append("")
    L.append("| fold | 유형 | 진폭군 | OC-SVM·band | IF·band | 제안 Fisher | 제안 S_amp | 제안 S_shape |")
    L.append("|---|---|---|---|---|---|---|---|")
    for split in SPLITS:
        for lono in LONOS:
            k = (split, lono)
            if k not in prop:
                continue
            ftype = "comp" if SPLIT_INFO[split][1] == "compositional" else "zero"
            grp = "high" if LONO_TARGET[lono] in HIGH_AMP else "low"
            L.append(f"| {split}_L{lono} | {ftype} | {grp} | "
                     f"{fmt(ocs.get(k, {}).get('auroc'))} | {fmt(iff.get(k, {}).get('auroc'))} | "
                     f"{fmt(prop[k]['fisher']['auroc'])} | {fmt(prop[k]['amp']['auroc'])} | "
                     f"{fmt(prop[k]['shape']['auroc'])} |")
    L.append("")

    # ② 밀도 모델 비교 (같은 6-band): S_amp vs OC-SVM·band vs IF·band
    fa_amp = {k: prop[k]["amp"]["auroc"] for k in prop}
    fa_ocs = {k: ocs[k]["auroc"] for k in ocs}
    fa_iff = {k: iff[k]["auroc"] for k in iff}
    L.append("## ② 밀도 모델 비교 — 같은 6-band 특징, S_amp(조건부 Gaussian) vs generic OCC")
    L.append("")
    L.append(f"overall(24 fold 평균): S_amp {fmt(*_ms(list(fa_amp.values())))} · "
             f"OC-SVM·band {fmt(*_ms(list(fa_ocs.values())))} · IF·band {fmt(*_ms(list(fa_iff.values())))}")
    L.append("")
    L.append("### PU paired (24 fold, Wilcoxon+Holm) — 기준 = 제안 S_amp")
    L.append("| 비교 | Δ(대상−S_amp) | p(Holm) | 대상 우세/24 |")
    L.append("|---|---|---|---|")
    p_ocs = paired(fa_ocs, fa_amp); p_iff = paired(fa_iff, fa_amp)
    hp = holm([p_ocs["p"], p_iff["p"]])
    L.append(f"| OC-SVM·band vs S_amp | {p_ocs['mean_diff']:+.3f} | {fmt(hp[0])} | {p_ocs['n_pos']}/{p_ocs['n']} |")
    L.append(f"| IF·band vs S_amp | {p_iff['mean_diff']:+.3f} | {fmt(hp[1])} | {p_iff['n_pos']}/{p_iff['n']} |")
    L.append("")

    # UODS 100 split paired
    up_amp = uods_proposed_amp_splits()
    uocs = uods_combo_splits("ocsvm", "band"); uiff = uods_combo_splits("if", "band")
    ua = {k: up_amp[k]["overall"] for k in up_amp}
    uo = {k: uocs[k]["overall"] for k in uocs}
    ui = {k: uiff[k]["overall"] for k in uiff}
    L.append("### UODS paired (100 split, Wilcoxon) — 기준 = 제안 S_amp")
    L.append(f"overall: S_amp {fmt(*_ms(list(ua.values())))} · OC-SVM·band {fmt(*_ms(list(uo.values())))} · "
             f"IF·band {fmt(*_ms(list(ui.values())))}")
    L.append("| 비교 | Δ(대상−S_amp) | p | 대상 우세/100 |")
    L.append("|---|---|---|---|")
    upo = paired(uo, ua); upi = paired(ui, ua)
    L.append(f"| OC-SVM·band vs S_amp | {upo['mean_diff']:+.3f} | {fmt(upo['p'])} | {upo['n_pos']}/{upo['n']} |")
    L.append(f"| IF·band vs S_amp | {upi['mean_diff']:+.3f} | {fmt(upi['p'])} | {upi['n_pos']}/{upi['n']} |")
    L.append("")

    # ③ 저진폭 정상 FPR · ρ
    L.append("## ③ 저진폭 정상 FPR · ρ(RMS,score) (PU 24 fold 평균)")
    L.append("")
    L.append("| 모델 | 저진폭 정상 FPR | 고진폭 정상 FPR | ρ(RMS,score) |")
    L.append("|---|---|---|---|")
    L.append(f"| S_amp | {fmt(_mean([prop[k]['amp']['low_fpr'] for k in prop]))} | "
             f"{fmt(_mean([prop[k]['amp']['high_fpr'] for k in prop]))} | "
             f"{fmt(_mean([prop[k]['amp']['rho'] for k in prop]))} |")
    L.append(f"| OC-SVM·band | {fmt(_mean([ocs[k]['low_fpr'] for k in ocs]))} | "
             f"{fmt(_mean([ocs[k]['high_fpr'] for k in ocs]))} | {fmt(_mean([ocs[k]['rho'] for k in ocs]))} |")
    L.append(f"| IF·band | {fmt(_mean([iff[k]['low_fpr'] for k in iff]))} | "
             f"{fmt(_mean([iff[k]['high_fpr'] for k in iff]))} | {fmt(_mean([iff[k]['rho'] for k in iff]))} |")
    L.append(f"| 제안 Fisher (참고) | {fmt(_mean([prop[k]['fisher']['low_fpr'] for k in prop]))} | "
             f"{fmt(_mean([prop[k]['fisher']['high_fpr'] for k in prop]))} | "
             f"{fmt(_mean([prop[k]['fisher']['rho'] for k in prop]))} |")
    L.append("")

    # 해석 [확정]/[가설] — 데이터 기반 자동 문장
    amp_ov = _ms(list(fa_amp.values()))[0]; ocs_ov = _ms(list(fa_ocs.values()))[0]
    L.append("## 해석")
    sig_ocs = (p_ocs["p"] is not None and hp[0] is not None and hp[0] < 0.05)
    if ocs_ov > amp_ov and sig_ocs:
        L.append(f"- **[확정]** 같은 6-band 특징에서 **OC-SVM·band가 제안 S_amp를 유의하게 상회**"
                 f"(Δ{p_ocs['mean_diff']:+.3f}, Holm p={fmt(hp[0])}, {p_ocs['n_pos']}/{p_ocs['n']}) "
                 f"→ 제안의 조건부 Gaussian amp head는 **generic OCC 대비 순이득 없음**. 이득의 원천은 특징(log-band)이지 amp 밀도 모델이 아님.")
    else:
        L.append(f"- **[관측]** S_amp overall {amp_ov:.3f} vs OC-SVM·band {ocs_ov:.3f} (Δ{p_ocs['mean_diff']:+.3f}, Holm p={fmt(hp[0])}). 상세 표 참조.")
    L.append("- **[확정]** raw 입력 대비 6-band 입력이 amp·shape 결함 동시 보존(B-1 표) — 표현이 판별력의 핵심.")
    L.append("- **[가설]** 제안 Fisher가 S_amp 단독·OCC 대비 갖는 이점은 shape branch 결합(dual)에서 오며, amp 밀도 모델링 자체의 기여는 제한적. (S_shape·Fisher 열/③ FPR로 뒷받침 정도 판단.)")
    L.append("")
    L.append("> 모델·하이퍼 변경 없음. 전 수치는 동결 캐시(제안 p2_log·UODS proposed, B-1 band)에서 재계산.")

    with open(report, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {report}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
