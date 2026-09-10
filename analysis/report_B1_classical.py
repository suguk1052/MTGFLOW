"""B-1 고전 baseline 리포트 — IF·OC-SVM × raw·band 를 Table 4 전 열로 집계 + proposed/raw paired.

입력: results/{Paderborn,UODS}/b1_{if,ocsvm}_{raw,band}_window_scores/*.npz (dump_B1_* 산출).
집계·지표 정의는 제안 파이프라인과 동일 함수를 재사용:
  PU  : diagnose_G3a_amp_bands._score_block (AUROC/per-fault/ρ/진폭군 FPR) + 24-fold paired Wilcoxon+Holm.
  UODS: diagnose_uods_fusion.subgroup_aurocs / fpr_at_val95 (§1·§2 열) + 100-split paired.
proposed/raw 참조값도 기존 캐시에서 재계산해 같은 표에 함께 싣는다.
  PU  proposed(Fisher-tail) = diag_P2_log_tail_fusion/*.json 의 auroc.fusion_B(5seed평균)
  PU  raw                    = b3_raw_window_scores/*.npz 의 te_f AUROC
  UODS proposed/raw          = uods_window_scores 캐시에 evaluate_run 적용(fisher/raw)

산출: reports/report_B1_classical.md
사용: conda run -n mtgflow python analysis/report_B1_classical.py
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from diagnose_G3a_amp_bands import (  # noqa: E402
    _score_block, safe_auroc, HIGH_AMP, SPLIT_INFO, LONO_TARGET,
    AMP_SENSITIVE, SHAPE_SENSITIVE,
)
from diagnose_uods_fusion import subgroup_aurocs, fpr_at_val95, evaluate_run  # noqa: E402

RES_PU = os.path.join(PROJECT_ROOT, "results", "Paderborn")
RES_UODS = os.path.join(PROJECT_ROOT, "results", "UODS")
UODS_CACHE = os.path.join(RES_UODS, "uods_window_scores")
SPLITS = ["123to0", "023to1", "013to2", "012to3"]
LONOS = [1, 2, 3, 4, 5, 6]
SEEDS = [2024, 2025, 2026, 2027, 2028]
COMBOS = [("if", "raw"), ("if", "band"), ("ocsvm", "raw"), ("ocsvm", "band")]
COMBO_LABEL = {("if", "raw"): "IF · raw", ("if", "band"): "IF · 6-log-band",
               ("ocsvm", "raw"): "OC-SVM · raw", ("ocsvm", "band"): "OC-SVM · 6-log-band"}


# ---------------------------------------------------------------------------
def _mean(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and x == x]
    return float(np.mean(v)) if v else float("nan")


def _ms(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and x == x]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def fmt(m, s=None, p=3):
    if m != m:
        return "—"
    return f"{m:.{p}f}" if s is None else f"{m:.{p}f}±{s:.{p}f}"


def holm(pvals):
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx); adj = [None] * len(pvals); prev = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvals[j])):
        a = min(1.0, pvals[i] * (m - rank)); a = max(a, prev); prev = a; adj[i] = a
    return adj


def paired(fold_a, fold_b):
    common = sorted(set(fold_a) & set(fold_b))
    xa = np.array([fold_a[k] for k in common]); xb = np.array([fold_b[k] for k in common])
    d = xa - xb
    r = {"n": len(common), "mean_diff": float(d.mean()) if len(d) else float("nan"),
         "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "p": None}
    if _HAVE_SCIPY and len(common) >= 1 and np.any(d != 0):
        try:
            r["p"] = float(wilcoxon(xa, xb).pvalue)
        except Exception:
            r["p"] = None
    return r


# ---------------------------------------------------------------------------
# PU
# ---------------------------------------------------------------------------
def _pu_block_from_npz(path):
    z = np.load(path, allow_pickle=True)
    te_f, va_f, tr_f = z["te_f"], z["va_f"], z["tr_f"]
    te_lab, te_ids, te_rms = z["te_lab"], z["te_ids"], z["te_rms"]
    va_ids, va_rms, tr_ids, tr_rms = z["va_ids"], z["va_rms"], z["tr_ids"], z["tr_rms"]
    tgt = te_lab == 0
    norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]])
    norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])
    return _score_block("b1", te_f, va_f, te_lab, te_ids, te_rms, tr_f, va_f,
                        te_rms[tgt], norm_rms, norm_ids, 95.0)


def pu_combo_folds(model, inp, out_root=RES_PU):
    """combo별 fold→평균 지표(seed평균). IF=5seed, OCSVM=1(s2024)."""
    d = os.path.join(out_root, f"b1_{model}_{inp}_window_scores")
    seeds = SEEDS if model == "if" else [2024]
    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            blks = []
            for s in seeds:
                p = os.path.join(d, f"{split}_LONO{lono}_s{s}.npz")
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


def pu_proposed_folds():
    d = os.path.join(RES_PU, "diag_P2_log_tail_fusion")
    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            vs = []
            for s in SEEDS:
                p = os.path.join(d, f"{split}_LONO{lono}_s{s}.json")
                if os.path.exists(p):
                    j = json.load(open(p))
                    v = j.get("auroc", {}).get("fusion_B")
                    if v is not None:
                        vs.append(v)
            if vs:
                folds[(split, lono)] = float(np.mean(vs))
    return folds


def pu_raw_folds():
    d = os.path.join(RES_PU, "b3_raw_window_scores")
    folds = {}
    for split in SPLITS:
        for lono in LONOS:
            vs = []
            for s in SEEDS:
                p = os.path.join(d, f"{split}_LONO{lono}_s{s}.npz")
                if os.path.exists(p):
                    z = np.load(p, allow_pickle=True)
                    vs.append(safe_auroc(z["te_lab"], z["te_f"]))
            if vs:
                folds[(split, lono)] = float(np.mean(vs))
    return folds


def pu_columns(folds):
    """fold dict(위 combo_folds 형식) → Table 4 열."""
    keys = list(folds)
    zero = [folds[k]["auroc"] for k in keys if SPLIT_INFO[k[0]][1] == "zero-support"]
    comp = [folds[k]["auroc"] for k in keys if SPLIT_INFO[k[0]][1] == "compositional"]
    # per-fault 집계 → amp/shape-sensitive 군 평균
    fa = {}
    for k in keys:
        for fid, v in folds[k]["per_fault"].items():
            fa.setdefault(fid, []).append(v)
    fa = {fid: _mean(vs) for fid, vs in fa.items()}
    amp_s = _mean([fa[f] for f in fa if f in AMP_SENSITIVE])
    shape_s = _mean([fa[f] for f in fa if f in SHAPE_SENSITIVE])
    return {
        "overall": _ms([folds[k]["auroc"] for k in keys]),
        "zero": _ms(zero), "comp": _ms(comp),
        "amp_sensitive": (amp_s, None), "shape_sensitive": (shape_s, None),
        "high_fpr": _ms([folds[k]["high_fpr"] for k in keys]),
        "low_fpr": _ms([folds[k]["low_fpr"] for k in keys]),
        "rho": _ms([folds[k]["rho"] for k in keys]),
    }


# ---------------------------------------------------------------------------
# UODS
# ---------------------------------------------------------------------------
UODS_SUBGROUPS = ["overall", "developing", "faulty", "nonball_developing", "nonball_faulty",
                  "fam_inner", "fam_outer", "fam_ball", "fam_cage", "ball", "nonball"]
EVAL_RUNS = list(range(5, 105))


def uods_combo_splits(model, inp, out_root=RES_UODS):
    d = os.path.join(out_root, f"b1_{model}_{inp}_window_scores")
    seeds = SEEDS if model == "if" else [2024]
    per_split = {}
    for r in EVAL_RUNS:
        sgs, fprs = [], []
        for s in seeds:
            p = os.path.join(d, f"eval_run{r}_s{s}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            sub = subgroup_aurocs(z["te_raw"], z["te_state"], z["te_fam"], z["te_isball"])
            fpr, _ = fpr_at_val95(z["te_raw"][np.asarray(z["te_state"], int) == 0], z["va_raw"])
            sgs.append(sub); fprs.append(fpr)
        if sgs:
            per_split[r] = {sg: _mean([x.get(sg) for x in sgs]) for sg in UODS_SUBGROUPS}
            per_split[r]["fpr_val95"] = _mean(fprs)
    return per_split


def uods_ref_splits(method):
    """proposed(fisher)/raw per-split overall (+subgroups) — uods_window_scores 캐시 재계산."""
    per_split = {}
    for r in EVAL_RUNS:
        sgs, fprs = [], []
        for s in SEEDS:
            pc = os.path.join(UODS_CACHE, f"uods_eval_run{r}_proposed_s{s}.npz")
            rc = os.path.join(UODS_CACHE, f"uods_eval_run{r}_raw_s{s}.npz")
            if not (os.path.exists(pc) and os.path.exists(rc)):
                continue
            res = evaluate_run(pc, rc)["methods"][method]
            sgs.append(res["subgroup"]); fprs.append(res["fpr_val95"])
        if sgs:
            per_split[r] = {sg: _mean([x.get(sg) for x in sgs]) for sg in UODS_SUBGROUPS}
            per_split[r]["fpr_val95"] = _mean(fprs)
    return per_split


def uods_columns(per_split):
    keys = list(per_split)
    cols = {sg: _ms([per_split[k][sg] for k in keys]) for sg in UODS_SUBGROUPS}
    cols["fpr_val95"] = _ms([per_split[k]["fpr_val95"] for k in keys])
    return cols


# ---------------------------------------------------------------------------
def build_report(out_root_pu=RES_PU, out_root_uods=RES_UODS):
    L = ["# B-1. 고전 baseline (Isolation Forest · OC-SVM) — Table 4 보강", ""]
    L.append("입력 (a) raw 2048 window(train-normal StandardScaler) / (b) 6-log-band log-RMS z-vector"
             "(제안 amp target과 동일: 경계 [1,3,10,32,102,323,1025], train-normal z-score).")
    L.append("부호 규약: anomaly = −(IF score_samples / OC-SVM decision_function). threshold = val-normal 95pct.")
    L.append("IF = seed 2024~2028 5-seed 평균, OC-SVM = 결정론 1회(raw는 N=10,000 서브샘플 fit).")
    L.append("")

    # ---- PU ----
    proposed = pu_proposed_folds(); raw = pu_raw_folds()
    L.append("## PU (24 fold) — Table 4 열")
    L.append("")
    L.append("| 모델·입력 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    combo_folds = {}
    for model, inp in COMBOS:
        f = pu_combo_folds(model, inp, out_root_pu); combo_folds[(model, inp)] = f
        if not f:
            L.append(f"| {COMBO_LABEL[(model, inp)]} | (결과 없음) | | | | | | | |"); continue
        c = pu_columns(f)
        L.append(f"| {COMBO_LABEL[(model, inp)]} | {fmt(*c['overall'])} | {fmt(*c['zero'])} | "
                 f"{fmt(*c['comp'])} | {fmt(c['amp_sensitive'][0])} | {fmt(c['shape_sensitive'][0])} | "
                 f"{fmt(*c['high_fpr'])} | {fmt(*c['low_fpr'])} | {fmt(*c['rho'])} |")
    # 참조행
    if raw:
        L.append(f"| raw MTGFlow (참조) | {fmt(*_ms(list(raw.values())))} | | | | | | | |")
    if proposed:
        L.append(f"| 제안 N6+log+Fisher (참조) | {fmt(*_ms(list(proposed.values())))} | | | | | | | |")
    L.append("")

    # paired
    L.append("### PU paired (24 fold overall AUROC, Wilcoxon + Holm)")
    L.append("")
    L.append("| 모델·입력 | vs raw: Δ | p(Holm) | 개선/24 | vs proposed: Δ | p(Holm) | 개선/24 |")
    L.append("|---|---|---|---|---|---|---|")
    pv_raw, pv_prop, rows = [], [], []
    for model, inp in COMBOS:
        f = combo_folds[(model, inp)]
        if not f:
            rows.append((COMBO_LABEL[(model, inp)], None, None)); pv_raw.append(None); pv_prop.append(None); continue
        fa = {k: f[k]["auroc"] for k in f}
        pr = paired(fa, raw) if raw else None
        pp = paired(fa, proposed) if proposed else None
        rows.append((COMBO_LABEL[(model, inp)], pr, pp))
        pv_raw.append(pr["p"] if pr else None); pv_prop.append(pp["p"] if pp else None)
    hr = holm(pv_raw); hp = holm(pv_prop)
    for (lbl, pr, pp), hrv, hpv in zip(rows, hr, hp):
        if pr is None:
            L.append(f"| {lbl} | — | — | — | — | — | — |"); continue
        L.append(f"| {lbl} | {pr['mean_diff']:+.3f} | {fmt(hrv)} | {pr['n_pos']}/{pr['n']} | "
                 f"{pp['mean_diff']:+.3f} | {fmt(hpv)} | {pp['n_pos']}/{pp['n']} |")
    L.append("")

    # ---- UODS ----
    L.append("## UODS (eval 100 split) — report_uods_full_eval §1·§2 열")
    L.append("")
    hdr = "| 모델·입력 | " + " | ".join(UODS_SUBGROUPS) + " | fpr_val95 |"
    L.append(hdr); L.append("|---|" + "---|" * (len(UODS_SUBGROUPS) + 1))
    uods_folds = {}
    for model, inp in COMBOS:
        ps = uods_combo_splits(model, inp, out_root_uods); uods_folds[(model, inp)] = ps
        if not ps:
            L.append(f"| {COMBO_LABEL[(model, inp)]} | " + " | ".join(["(없음)"] * (len(UODS_SUBGROUPS) + 1)) + " |")
            continue
        c = uods_columns(ps)
        L.append(f"| {COMBO_LABEL[(model, inp)]} | " +
                 " | ".join(fmt(*c[sg]) for sg in UODS_SUBGROUPS) + f" | {fmt(*c['fpr_val95'])} |")
    # 참조행 proposed/raw
    uods_ref = {}
    for m in ("raw", "fisher"):
        try:
            uods_ref[m] = uods_ref_splits(m)
        except Exception as e:
            uods_ref[m] = {}; L.append(f"<!-- UODS ref {m} 계산 실패: {e} -->")
    for m, lbl in (("raw", "raw MTGFlow (참조)"), ("fisher", "제안 Fisher-tail (참조)")):
        if uods_ref.get(m):
            c = uods_columns(uods_ref[m])
            L.append(f"| {lbl} | " + " | ".join(fmt(*c[sg]) for sg in UODS_SUBGROUPS) + f" | {fmt(*c['fpr_val95'])} |")
    L.append("")

    # UODS paired (overall)
    L.append("### UODS paired (100 split overall AUROC, Wilcoxon)")
    L.append("")
    L.append("| 모델·입력 | vs raw: Δ | p | 개선/100 | vs proposed: Δ | p | 개선/100 |")
    L.append("|---|---|---|---|---|---|---|")
    for model, inp in COMBOS:
        ps = uods_folds[(model, inp)]
        if not ps:
            L.append(f"| {COMBO_LABEL[(model, inp)]} | — | — | — | — | — | — |"); continue
        fa = {k: ps[k]["overall"] for k in ps}
        rr = uods_ref.get("raw", {}); pp = uods_ref.get("fisher", {})
        r1 = paired(fa, {k: rr[k]["overall"] for k in rr}) if rr else None
        r2 = paired(fa, {k: pp[k]["overall"] for k in pp}) if pp else None
        c1 = f"{r1['mean_diff']:+.3f} | {fmt(r1['p'])} | {r1['n_pos']}/{r1['n']}" if r1 else "— | — | —"
        c2 = f"{r2['mean_diff']:+.3f} | {fmt(r2['p'])} | {r2['n_pos']}/{r2['n']}" if r2 else "— | — | —"
        L.append(f"| {COMBO_LABEL[(model, inp)]} | {c1} | {c2} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root_pu", default=RES_PU)
    ap.add_argument("--out_root_uods", default=RES_UODS)
    ap.add_argument("--report", default=os.path.join(PROJECT_ROOT, "reports", "report_B1_classical.md"))
    args = ap.parse_args()
    md = build_report(args.out_root_pu, args.out_root_uods)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        f.write(md)
    print(f"wrote {args.report}\n")
    print(md[:2000])


if __name__ == "__main__":
    main()
