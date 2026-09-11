"""B 트랙 최종 통합 Table 4 — 전 baseline 행을 한 표로 (analysis-only).

PU Table 4: IF·raw/IF·band/OC-SVM·raw/OC-SVM·band(B-1) · Deep SVDD(B-2) · CATCH(B-3) · KAN-AD(B-4)
            · raw MTGFlow · 제안 N6+log+Fisher. 전 열 + 제안 대비 24 fold paired(Wilcoxon+Holm).
UODS 표: UODS 있는 행만(IF/OC-SVM×raw/band·raw MTGFlow·제안) §1·§2 열. PU 전용 행은 —(PU 전용).

집계 함수는 report_B1/B2/B3/B4에서 재사용. 제안/raw 전 열은 캐시(p2_log/b3)에서 _score_block으로 재계산.
산출: reports/report_B_external_baselines.md
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
    pu_combo_folds, pu_columns, paired, holm, fmt, _mean, _ms,
    uods_combo_splits, uods_columns, uods_ref_splits, UODS_SUBGROUPS,
    RES_PU, SPLITS, LONOS, SEEDS, _pu_block_from_npz,
)
from report_B2_deepsvdd import b2_folds  # noqa: E402
from report_B4_kanad import b4_folds  # noqa: E402
from report_B3_catch import fold_full_metrics as catch_fold_metrics  # noqa: E402
from diagnose_G5_tail_fusion import fisher_tail  # noqa: E402
from diagnose_G3a_amp_bands import _score_block, SPLIT_INFO  # noqa: E402

P2_LOG = os.path.join(RES_PU, "p2_log_window_scores")
B3_RAW = os.path.join(RES_PU, "b3_raw_window_scores")


def _agg_block(blks):
    return {"auroc": _mean([b["auroc"] for b in blks]),
            "rho": _mean([b["rho_rms_score"] for b in blks]),
            "high_fpr": _mean([b["high_amp_fpr"] for b in blks]),
            "low_fpr": _mean([b["low_amp_fpr"] for b in blks]),
            "per_fault": {fid: _mean([b["per_fault_auroc"].get(fid, {}).get("auroc") for b in blks])
                          for fid in set().union(*[b["per_fault_auroc"].keys() for b in blks])}}


def raw_full_folds():
    """raw MTGFlow 전 열 (b3_raw npz te_f)."""
    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            blks = [_pu_block_from_npz(p) for s in SEEDS
                    if os.path.exists(p := os.path.join(B3_RAW, f"{split}_LONO{lono}_s{s}.npz"))]
            if blks:
                folds[(split, lono)] = _agg_block(blks)
    return folds


def proposed_full_folds():
    """제안 Fisher-tail 전 열 (p2_log npz에서 fisher_tail + _score_block)."""
    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            blks = []
            for s in SEEDS:
                p = os.path.join(P2_LOG, f"{split}_LONO{lono}_s{s}.npz")
                if not os.path.exists(p):
                    continue
                z = np.load(p, allow_pickle=True)
                te_sh, te_am, va_sh, va_am, tr_sh, tr_am = (z["te_sh"], z["te_am"], z["va_sh"], z["va_am"], z["tr_sh"], z["tr_am"])
                te_lab, te_ids, te_rms = z["te_lab"], z["te_ids"], z["te_rms"]
                va_ids, va_rms, tr_ids, tr_rms = z["va_ids"], z["va_rms"], z["tr_ids"], z["tr_rms"]
                fte = fisher_tail(te_sh, te_am, va_sh, va_am)
                fva = fisher_tail(va_sh, va_am, va_sh, va_am)
                ftr = fisher_tail(tr_sh, tr_am, va_sh, va_am)
                tgt = te_lab == 0
                norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]])
                norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])
                blks.append(_score_block("prop", fte, fva, te_lab, te_ids, te_rms, ftr, fva,
                                         te_rms[tgt], norm_rms, norm_ids, 95.0))
            if blks:
                folds[(split, lono)] = _agg_block(blks)
    return folds


def catch_row():
    """CATCH 전 열 (fold_full_metrics 재실행: CATCH 로더 복원)."""
    per = {}
    for split in SPLITS:
        for lono in LONOS:
            r = catch_fold_metrics(split, lono)
            if r is not None:
                per[(split, lono)] = r
    keys = list(per)
    zero = [per[k]["auroc"] for k in keys if SPLIT_INFO[k[0]][1] == "zero-support"]
    comp = [per[k]["auroc"] for k in keys if SPLIT_INFO[k[0]][1] == "compositional"]
    return {"overall": _ms([per[k]["auroc"] for k in keys]), "zero": _ms(zero), "comp": _ms(comp),
            "amp_sensitive": (_mean([per[k]["amp_sensitive"] for k in keys]), None),
            "shape_sensitive": (_mean([per[k]["shape_sensitive"] for k in keys]), None),
            "high_fpr": _ms([per[k]["high_fpr"] for k in keys]),
            "low_fpr": _ms([per[k]["low_fpr"] for k in keys]),
            "rho": _ms([per[k]["rho"] for k in keys])}, {k: per[k]["auroc"] for k in keys}


def _row(label, cols, fa, prop_fa, note=""):
    """cols=pu_columns dict, fa=fold auroc dict(paired용, None이면 생략)."""
    line = (f"| {label} | {fmt(*cols['overall'])} | {fmt(*cols['zero'])} | {fmt(*cols['comp'])} | "
            f"{fmt(cols['amp_sensitive'][0])} | {fmt(cols['shape_sensitive'][0])} | "
            f"{fmt(*cols['high_fpr'])} | {fmt(*cols['low_fpr'])} | {fmt(*cols['rho'])} |")
    pcell = "—"
    if fa is not None and prop_fa is not None:
        pp = paired(fa, prop_fa)
        pcell = f"{pp['mean_diff']:+.3f} (p={fmt(pp['p'])}, {pp['n_pos']}/{pp['n']})"
    return line + f" {pcell} |", (paired(fa, prop_fa)["p"] if (fa is not None and prop_fa is not None) else None)


def main():
    report = os.path.join(PROJECT_ROOT, "reports", "report_B_external_baselines.md")

    # --- PU 각 모델 folds/cols ---
    combos = {
        "IF · raw": pu_combo_folds("if", "raw"), "IF · 6-log-band": pu_combo_folds("if", "band"),
        "OC-SVM · raw": pu_combo_folds("ocsvm", "raw"), "OC-SVM · 6-log-band": pu_combo_folds("ocsvm", "band"),
        "Deep SVDD (raw)": b2_folds(), "KAN-AD (raw, 참고)": b4_folds(),
    }
    raw_f = raw_full_folds(); prop_f = proposed_full_folds()
    prop_fa = {k: prop_f[k]["auroc"] for k in prop_f}

    L = ["# B 트랙 — 외부 baseline Table 4 (최종 통합)", ""]
    L.append("PU 24 fold(seed 평균)·동일 split·window·scaler·threshold. proposed = N6+log+Fisher-tail(동결). "
             "paired = 각 행 vs 제안, 24 fold Wilcoxon(p=Holm 미적용 단일비교 raw p). GPU=Deep SVDD·KAN-AD, 나머지 CPU.")
    L.append("")
    L.append("## PU Table 4")
    L.append("")
    L.append("| 모델 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) | vs 제안 Δ(p,우세/24) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    pvals, rows = [], []
    for label, f in combos.items():
        cols = pu_columns(f)
        fa = {k: f[k]["auroc"] for k in f}
        line, pv = _row(label, cols, fa, prop_fa)
        rows.append(line); pvals.append(pv)
    # CATCH (특수 집계)
    catch_cols, catch_fa = catch_row()
    cline, cpv = _row("CATCH (재구성 TSAD, 참고)", catch_cols, catch_fa, prop_fa)
    rows.append(cline); pvals.append(cpv)
    # raw / 제안
    raw_cols = pu_columns(raw_f); raw_fa = {k: raw_f[k]["auroc"] for k in raw_f}
    rline, rpv = _row("raw MTGFlow", raw_cols, raw_fa, prop_fa); rows.append(rline); pvals.append(rpv)
    prop_cols = pu_columns(prop_f)
    pline, _ = _row("**제안 N6+log+Fisher**", prop_cols, None, None); rows.append(pline)
    L.extend(rows)
    L.append("")
    L.append("- **핵심[확정]**: 6-log-band 특징을 쓴 고전 OCC(OC-SVM·band 0.933·IF·band 0.882)가 제안(0.877)에 필적/상회. "
             "raw 입력(IF·OC-SVM·Deep SVDD·raw MTGFlow·KAN-AD)은 상대적으로 약함. → 이득의 원천 = log-band 표현(B-5 상술).")
    L.append("- **프레이밍(B-G2)**: '우세' 주장은 OCC 계열 한정. CATCH·KAN-AD는 표준 TSAD 참고 행 "
             "(① 시점탐지 설계 ② K=1 채널 융합 비활성 ③ seq_len 2048 제약). CATCH amp/shape·FPR·ρ는 CATCH 로더 재실행 복원값(train score 부재로 정상 pool=val+test-normal).")
    L.append("")

    # --- UODS 표 (UODS 있는 행만) ---
    L.append("## UODS Table (eval 100 split, seed 평균) — report_uods_full_eval §1·§2 열")
    L.append("")
    hdr = "| 모델 | " + " | ".join(UODS_SUBGROUPS) + " | fpr_val95 |"
    L.append(hdr); L.append("|---|" + "---|" * (len(UODS_SUBGROUPS) + 1))
    ucombos = {"IF · raw": ("if", "raw"), "IF · 6-log-band": ("if", "band"),
               "OC-SVM · raw": ("ocsvm", "raw"), "OC-SVM · 6-log-band": ("ocsvm", "band")}
    for label, (m, i) in ucombos.items():
        ps = uods_combo_splits(m, i)
        if not ps:
            continue
        c = uods_columns(ps)
        L.append(f"| {label} | " + " | ".join(fmt(*c[sg]) for sg in UODS_SUBGROUPS) + f" | {fmt(*c['fpr_val95'])} |")
    for m, lbl in (("raw", "raw MTGFlow"), ("fisher", "**제안 Fisher-tail**")):
        try:
            c = uods_columns(uods_ref_splits(m))
            L.append(f"| {lbl} | " + " | ".join(fmt(*c[sg]) for sg in UODS_SUBGROUPS) + f" | {fmt(*c['fpr_val95'])} |")
        except Exception as e:
            L.append(f"| {lbl} | (계산 실패: {e}) |")
    L.append(f"| Deep SVDD / CATCH / KAN-AD | " + " | ".join(["—(PU 전용)"] * (len(UODS_SUBGROUPS) + 1)) + " |")
    L.append("")
    L.append("> 상세 per-모델 리포트: report_B1_classical / report_B2_deepsvdd / report_B3_catch / report_B4_kanad / report_B5_feature_vs_model.")

    with open(report, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {report}\n"); print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
