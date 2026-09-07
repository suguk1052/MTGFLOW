"""P-1 Band count sweep — N별 Fusion(equal-z A / Fisher-tail B) 집계 및 N=ref paired 비교.

diagnose_G5_tail_fusion.py 가 각 N에 대해 남긴 seed별 aggregate_G5_s<seed>.json + per-fold JSON을
읽어(학습·추론 없음, analysis-only) N=ref(기본 6=g3a) 대비 paired 비교를 만든다.

diag 경로 규약:
  N=6  → results/Paderborn/diag_G5_tail_fusion (기존 G-5 재사용)
  그 외 → results/Paderborn/diag_P1_n{N}_tail_fusion

비교(각 N, ref 대비):
  - 전체 Fisher(B)·equal-z(A) AUROC mean±std, seed별
  - zero-support / compositional (B·A)
  - amp-sensitive / shape-sensitive fault군 (B·A)
  - 저·고진폭 정상 FPR (B·A)
  - 24-fold paired Wilcoxon (B, A) : 각 fold seed평균으로 pairing → N vs ref

산출: reports/report_P1_fisher_<tag>.md  (+ stdout 요약)
"""
import argparse
import glob
import json
import os
import re

import numpy as np

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJECT_ROOT, "results", "Paderborn")
SPLITS = ["123to0", "023to1", "013to2", "012to3"]
LONOS = [1, 2, 3, 4, 5, 6]


def diag_dir(N):
    return os.path.join(RES, "diag_G5_tail_fusion") if N == 6 else os.path.join(RES, f"diag_P1_n{N}_tail_fusion")


def ms(vals):
    v = [x for x in vals if x is not None and isinstance(x, (int, float)) and x == x]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def fmt(m, s, p=3):
    return "nan" if m != m else f"{m:.{p}f}±{s:.{p}f}"


def gpath(a, *keys):
    for k in keys:
        a = a.get(k) if isinstance(a, dict) else None
        if a is None:
            return None
    return a


def load_seed_aggs(N, seeds):
    """seed별 aggregate JSON → list. 없으면 폴더 내 전부."""
    d = diag_dir(N)
    aggs, used = [], []
    for s in seeds:
        p = os.path.join(d, f"aggregate_G5_s{s}.json")
        if os.path.exists(p):
            aggs.append(json.load(open(p)))
            used.append(s)
    return aggs, used


def per_fold_metric(N, seeds, method):
    """(split,lono) → 해당 fold의 seed평균 AUROC(method). paired 비교용. 24 fold dict 반환."""
    d = diag_dir(N)
    fold_vals = {}
    for split in SPLITS:
        for lono in LONOS:
            vs = []
            for s in seeds:
                p = os.path.join(d, f"{split}_LONO{lono}_s{s}.json")
                if not os.path.exists(p):
                    continue
                j = json.load(open(p))
                v = gpath(j, "auroc", method)
                if v is not None:
                    vs.append(v)
            if vs:
                fold_vals[(split, lono)] = float(np.mean(vs))
    return fold_vals


def seed_overall(N, seeds, method):
    aggs, used = load_seed_aggs(N, seeds)
    return {s: gpath(a, "overall", method) for s, a in zip(used, aggs)}


def agg_row(N, seeds, path_keys):
    """seed aggregate에서 path_keys 경로 값을 method별로 mean±std."""
    aggs, _ = load_seed_aggs(N, seeds)
    out = {}
    for method in ("fusion_A", "fusion_B", "raw"):
        vals = [gpath(a, *[k if k != "<m>" else method for k in path_keys]) for a in aggs]
        out[method] = ms(vals)
    return out


def paired(N, ref, seeds, method):
    a = per_fold_metric(N, seeds, method)
    b = per_fold_metric(ref, seeds, method)
    common = sorted(set(a) & set(b))
    xa = np.array([a[k] for k in common])
    xb = np.array([b[k] for k in common])
    d = xa - xb
    res = {"n": len(common), "mean_diff": float(d.mean()) if len(d) else float("nan"),
           "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "p": None}
    if _HAVE_SCIPY and len(common) >= 1 and np.any(d != 0):
        try:
            res["p"] = float(wilcoxon(xa, xb).pvalue)
        except Exception:
            res["p"] = None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", required=True, help="비교할 N 목록")
    ap.add_argument("--ref", type=int, default=6, help="paired 기준 N (기본 6=g3a)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    ns = args.ns
    ref = args.ref
    tag = args.tag or ("_".join(f"n{n}" for n in ns) + f"_vs_n{ref}")
    report_path = os.path.join(PROJECT_ROOT, "reports", f"report_P1_fisher_{tag}.md")

    L = ["# P-1 Band count — Fisher-tail(B)/equal-z(A) N별 집계 & paired 비교", ""]
    L.append(f"비교 N={ns}, paired 기준 N={ref}. seed {args.seeds}. analysis-only(캐시 재사용, 학습·추론 없음).")
    L.append(f"paired = 각 fold의 seed평균 AUROC로 24-fold pairing → Wilcoxon."
             + ("" if _HAVE_SCIPY else "  ⚠️ scipy 없음: p-value 생략, 부호수만."))
    L.append("")

    # seed별 완료 수 진단
    L.append("## 완료 상태")
    for n in ns + ([ref] if ref not in ns else []):
        aggs, used = load_seed_aggs(n, args.seeds)
        nfold = per_fold_metric(n, args.seeds, "fusion_B")
        L.append(f"- N={n}: seed aggregate {len(used)}개 {used}, per-fold(seed평균) {len(nfold)}/24")
    L.append("")

    METHS = [("fusion_B", "Fisher-tail(B)"), ("fusion_A", "equal-z(A)")]

    def block(title, path_keys):
        L.append(f"## {title}")
        L.append("")
        L.append("| N | " + " | ".join(lbl for _, lbl in METHS) + " |")
        L.append("|---|" + "---|" * len(METHS))
        for n in ns:
            row = agg_row(n, args.seeds, path_keys)
            cells = [fmt(*row[m]) for m, _ in METHS]
            mark = " (ref)" if n == ref else ""
            L.append(f"| {n}{mark} | " + " | ".join(cells) + " |")
        L.append("")

    block("전체 AUROC", ["overall", "<m>"])
    block("zero-support 외삽", ["by_fold_type", "zero-support", "auroc", "<m>"])
    block("compositional", ["by_fold_type", "compositional", "auroc", "<m>"])
    block("amp-sensitive fault", ["fault_group", "amp_sensitive", "<m>"])
    block("shape-sensitive fault", ["fault_group", "shape_sensitive", "<m>"])
    block("정상 FPR 고진폭", ["normal_fpr", "<m>", "high"])
    block("정상 FPR 저진폭", ["normal_fpr", "<m>", "low"])

    # seed별 overall
    L.append("## seed별 전체 AUROC (Fisher B / equal-z A)")
    L.append("")
    L.append("| N | " + " | ".join(f"s{s}" for s in args.seeds) + " |")
    L.append("|---|" + "---|" * len(args.seeds))
    for n in ns:
        ob = seed_overall(n, args.seeds, "fusion_B")
        oa = seed_overall(n, args.seeds, "fusion_A")
        cells = [f"B {ob.get(s):.3f}/A {oa.get(s):.3f}" if ob.get(s) is not None else "-" for s in args.seeds]
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("")

    # paired Wilcoxon vs ref (+ Holm-Bonferroni 다중비교 보정)
    def holm(pvals):
        """Holm-Bonferroni 보정. pvals: list(float|None) → adjusted(list). None은 보정 제외."""
        idx = [i for i, p in enumerate(pvals) if p is not None]
        m = len(idx)
        adj = [None] * len(pvals)
        prev = 0.0
        for rank, i in enumerate(sorted(idx, key=lambda j: pvals[j])):
            a = min(1.0, pvals[i] * (m - rank))
            a = max(a, prev)
            prev = a
            adj[i] = a
        return adj

    L.append(f"## 24-fold paired Wilcoxon signed-rank test (N vs ref N={ref})")
    L.append("")
    L.append(f"> 각 fold의 seed평균 Fisher(B)/equal-z(A) AUROC로 24-fold pairing 후 **Wilcoxon signed-rank test**"
             f"(정규성 무가정, 양측). mean_diff>0 이면 N이 ref보다 높음. "
             f"여러 N을 ref와 비교하므로 **Holm-Bonferroni 보정 p**(m=비교 N 수={len([n for n in ns if n!=ref])})도 병기. "
             f"탐색용 — N* 판정은 P-G2.")
    L.append("")
    for m, lbl in METHS:
        cand = [n for n in ns if n != ref]
        raws = [paired(n, ref, args.seeds, m) for n in cand]
        holm_p = holm([r["p"] for r in raws])
        L.append(f"### {lbl}")
        L.append("| N | n_fold | mean_diff(N−ref) | pos/neg | p(raw) | p(Holm) |")
        L.append("|---|---|---|---|---|---|")
        for n, r, hp in zip(cand, raws, holm_p):
            pstr = "nan" if r["p"] is None else f"{r['p']:.4f}"
            hstr = "nan" if hp is None else f"{hp:.4f}"
            L.append(f"| {n} | {r['n']} | {r['mean_diff']:+.4f} | {r['n_pos']}/{r['n_neg']} | {pstr} | {hstr} |")
        L.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {report_path}")
    for n in ns:
        ob = agg_row(n, args.seeds, ["overall", "<m>"])
        print(f"  N={n}: B={fmt(*ob['fusion_B'])}  A={fmt(*ob['fusion_A'])}")


if __name__ == "__main__":
    main()
